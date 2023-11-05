# -*- coding:utf8 -*-
"""

USAGE :

1/ aligning 2 files FILE1 and FILE2 : 

python3 ailign.py [--inputFormat INPUTFORMAT] --inputFile1 FILE1 --inputFile2 FILE2 --outputFilename OUTPUTFILENAME --outputFormats FORMATS

Examples :
python3 ailign.py --inputFormat json --inputFile1 4.\ stanza/KHM53.1819.grimm.de.json --inputFile2 4.\ stanza/KHM53.1869.alsleben.fr.json --outputFilename KHM53.alsleben.de-fr.txt --outputFormats txt ces
python3 ailign.py --inputFile1 2.\ txt/KHM53.1846.martin.fr.txt --inputFile2 2.\ txt/KHM53.1869.alsleben.fr.txt --outputFilename 5.\ aligned/KHM.1846-1869.fr-fr --outputFormats tmx txt  --savePlot --verbose
python3 ailign.py --inputFile1 corpus_aristophane/Plutus.Fleury.fr.txt --inputFile2 corpus_aristophane/Plutus.Fallex.fr.txt --outputFilename corpus_aristophane_aligné/Plutus.Fallex-Fleury.fr-fr --outputFormats tmx txt  --savePlot --verbose --margin 0.01 --cosThreshold 0.5 --k 2 --deltaX 20 --minDensityRatio 1.1


NB : 
- OUTPUTFILENAME is the file name without the extension. The extension will be added according to the format
- FORMATS may contain more than one format ex. "ces txt"


2/ aligning the files that matches PATTERN in INPUTDIR and write the output in OUTPUTDIR in the formats FORMATS :

python3 ailign.py --filePattern PATTERN --inputDir INPUTDIR --outputDir OUTPUTDIR --l1 LANG1 --l2 LANG2 --outputFormat FORMATS 

Options : 
  --writeAnchorPoints : to write the anchor points (incomplete but very reliable 1-1 alignement)
  --runDTW : to extract the complete alignment with 0-1, 1-0, 1-1, 1-2, 2-1, 1-3, 3-1, 2-2 grouping
                this extraction may be far longer !

How anchor points are filtered :
 
First of all, candidate points are created when a similarity threshold greater than cosThreshold (typically 0.4 or 0.5) is obtained between a sentence x and a sentence y. Then, for each column or row, only the k points with the highest scores are retained (the kBest parameter is set to a default value of 4).
Then, for each column or row, only the k points with the highest scores are retained (kBest parameter set to 4 by default).
At this stage, filtering is performed using the margin parameter, which allows us to retain only those points with a score greater than margin compared with their best competitor (margin=0.05 by default). If we apply this criterion, it may be consistent to set kBest to 1.

We then apply a two-stage high-pass filter.
The first filtering corresponds to the filterPoints() function. The first filter is based on a calculation of the density of candidates around each candidate point. This density is not calculated in a square centered around the point, but rather in a corridor centered on the diagonal passing through the point (the alignment path generally follows this diagonal). The width of this corridor corresponds to the deltaY parameter. The length of this corridor corresponds to the deltaY parameter. The number of candidate points divided by the size of this space gives a density value. If this density, divided by the average density of all candidate points, is greater than a certain ratio (minDensityRatio, typically around 0.5) then the point is retained. 
The second filter, which corresponds to the resolvingConflicts() function, focuses on resolving conflicts on the vertical and horizontal axes respectively - when for the same x-coordinate there are several points with different y-coordinates, and conversely, when for the same y-coordinate there are several points with different x-coordinates - these cases only arise if KBest is greater than 1. Competitors are eliminated on the basis of density: only the point with the best density along its diagonal is retained.
This density filtering can be repeated once if the --reiterateFiltering parameter is given.

"""

import os
import sys
import re
import argparse
import math
import array
import time
import json
import warnings
import xml.etree.ElementTree as ET
import shelve
import math

from BTrees.OOBTree import OOBTree

import numpy as np

import matplotlib.pyplot as plt

import torch



# reading the command line arguments
parser = argparse.ArgumentParser(
    prog='ailign',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='''\
A program that align sentences for parallel text pairs. 
    
Input : 
* --inputFile1 and --inputFile2 indicate a parallel text pair
* for multiple files aligning, when using --inputDir param, the corresponding files 
must be named following this pattern : NAME..*.L1.txt NAME..*.L2.txt
* alternatively, text pairs may be listed in a file with --inputFileList param
* Recognized format are TXT, ARC, JSON, TSV, XML-CONLL, XML
* In TXT format, files should be formatted as raw utf8 text with one sentence per line.

Output :
The output will yield a new file (or a new pair of files) that contain a selection of 
sentences in TXT, CES, TSV, BERTALIGN or ARC format, in order to be easily aligned
in a next step (the non parallel text is supposed to be removed).
'''
)

# main arguments for input / output
parser.add_argument('--l1', type=str, help='The source language (ISO : ex. "en" for English)', default='en')
parser.add_argument('--l2', type=str, help='The target language (ISO : ex. "fr" for French, "*" for any)', default='*')
parser.add_argument('-i','--inputFormat',help='Format of the input (txt, arc, ces, json, tsv, xml-conll, xml)',default="txt")
parser.add_argument('--xmlGuide',nargs='+',type=str,help='List of markups that should be read in the XML input',default=["speaker","p"])
parser.add_argument('--col1',help='For TSV format, indicate the column of l1', type=int, default=0)
parser.add_argument('--col2',help='For TSV format, indicate the column of l2', type=int, default=1)
parser.add_argument('-o','--outputFormats',nargs='+',type=str,help='Formats of the output (TXT, CES, ARC, XML, BERTALIGN)',default=["txt","tmx","ces"])
parser.add_argument('--inputFile1', type=str, help='The l1 input file to process', default='')
parser.add_argument('--inputFile2', type=str, help='The l2 input file to process', default='')
parser.add_argument('--inputFileList', type=str, help='A tsv file with corresponding filenames separated by tab', default='')
parser.add_argument('--inputDir', type=str, help='The directory to process', default='.')
parser.add_argument('--outputDir', type=str, help='The directory to save output files', default='.')
parser.add_argument('--outputFilename', type=str, help='The output filename (optional), without format extension', default='')
parser.add_argument('-f','--filePattern', type=str, help='The pattern of the files that should be processed. A capturing group such as (.*) should capture the common prefix between aligned files.', default=r'([^\\/]*)[.](\w\w\w?)[.]\w+$')
parser.add_argument('--writeAnchorPoints',help='Write anchor points',action="store_true",default=False)
parser.add_argument('--writeSegmentedInput',help='Write sentence segmented input files in txt format',action="store_true",default=False)
parser.add_argument('--writeIntervals',help='Write aligned intervals (as corresponding sentence numbers)',action="store_true",default=False)
parser.add_argument('--printIds',help='Print IDs in txt output',action="store_true",default=False)
parser.add_argument('--splitSent',help='Split the txt segments into sentences',action="store_true",default=False)
parser.add_argument('--useSentenceSegmenter',help='Use the trankit sentence segmenter for txt input (instead of regex segmenter)',action="store_true",default=False)
parser.add_argument('--mergeLines',help='Merge lines until a line ends with a separator for txt input',action="store_true",default=False)
parser.add_argument('--adaptativeMode',help='Using interval detection, compute estimated sentRatio and charRatio, and reiterate filtering.',action="store_true",default=False)

# special arguments for output control
parser.add_argument('-v','--verbose',help='Verbose messages',action="store_true")
parser.add_argument('-w','--writeAlignableArea',help='Write alignable area files',action="store_true",default=False)
parser.add_argument('-V','--veryVerbose',help='Very verbose messages',action="store_true")
parser.add_argument('--savePlot',help='Save scatter plot in a png file',action="store_true",default=False)
parser.add_argument('--showPlot',help='Show scatter plot (with a pause during execution)',action="store_true",default=False)

# controlling stage 1 and 2
parser.add_argument('--detectIntervals',help='Detect alignable interval using anchor points.',action="store_true",default=False)
parser.add_argument('-u','--useNgrams',help='Use ngrams to extract points',action="store_true",default=False)
parser.add_argument('-r','--doNotRunDTW', help='Perform only first step without DTW algorithm)',action="store_true",default=False)
parser.add_argument('--groupAfterOne2OnePairing', help='Run DTW algorithm with only 1-1 pairing, then, group the contiguous points',action="store_true",default=False)
parser.add_argument('--lateGrouping', help='Run DTW algorithm with only 1-1 pairing, then, group the contiguous points with lateGrouping method (greedy algorithm)',action="store_true",default=False)
parser.add_argument('--noMarginPenalty', help='Do not compute the similarity with neighbouring sentences, and substract the neighbouring similarity to the bead similarity)',action="store_true",default=False)
parser.add_argument('--deltaDist',type=float,help='The min difference between distances for grouping',default=0.05)


# controlling anchor points building and filtering 
# (important parameters are : cosThreshold, kBest, deltaX, minDensityRatio)
parser.add_argument('--embedModel',type=str,help='Choose embedding model : sbert or laser or labse-keras',default="sbert")
parser.add_argument('--modelName',type=str,help='Choose sbert model name (default=sentence-transformers/LaBSE)',default="sentence-transformers/LaBSE")
parser.add_argument('-l','--cosThreshold',type=float,help='The minimum similarity for labse vectors to yield one point',default=0.4)
parser.add_argument('--cosThresholdInOutputAnchors',type=float,help='The minimum similarity for final anchor points',default=0.5)
parser.add_argument('-n','--ngram', type=int, help='The ngram size', default=4)
parser.add_argument('-d','--diceThreshold', type=float, help='The minimum dice score to yield a candidate point', default=0.05)
parser.add_argument('--margin', type=float, help='Margin used to eliminate sentences that have too close neighbours on the vertical or horizontal axis', default=0.05)
parser.add_argument('-k','--kBest', type=int, help='Number of the best coordinates for each line ore column to keep when creating points', default=4)
parser.add_argument('-x','--deltaX', type=int, help='Local space definition : +/-delta X on horizontal axis', default=20)
parser.add_argument('-y','--deltaY', type=int, help='Local space definition : +/-delta Y on vertical axis', default=3)
parser.add_argument('-H','--minHorizontalDensity', type=float, help='The minimal horizontal density in a interval to be kept in the final result', default=0.05)
parser.add_argument('-m','--maxDistToTheDiagonal', type=int, help='The maximal distance to the diagonal (inside a given interval) for a point to be taken into account in the horizontal density', default=20)
parser.add_argument('-D','--minDensityRatio', type=float, help='The minimal local density ratio (reported to the average local density) to keep a candidate point', default=0.3)
parser.add_argument('-g','--maxGapSize', type=int, help='The maximal distance between to consecutive points in the same interval', default=100)
parser.add_argument('--diagBeam', type=float, help='A real number in the range 0-1 which indicate the max distance of anchor points to the diagonal (vertically), in proportion (1 indicates that the whole search space is used', default=1)
parser.add_argument('--localDiagBeam', type=float, help='A real number in the range 0-1 which indicate the max distance of anchor points to the diagonal of each alignable interval (vertically), in proportion (1 indicates that the whole search space is used', default=0.2)
parser.add_argument('--sentRatio', type=float, help='The sentence ratio is used during anchor point filtering. Normally computed automatically, may be forced when texts have very different length.', default=0)
parser.add_argument('--charRatio', type=float, help='The character ratio is used during final aligning when groups of sentences are paired. Normally computed automatically, may be forced when texts have very different length.', default=0)
parser.add_argument('--reiterateFiltering', help='Filter the anchor points according to density twice',action="store_true",default=False)

# controlling DTW algorithm
parser.add_argument('--dtwBeam', help='Max dist to the anchor point in DTW algorithm',type=int,default=3)
parser.add_argument('--localBeamDecay', help='Decreasing value of localBeam at each recursion step',type=float,default=0.5)
parser.add_argument('--distNull', help='Default distance for null correspondance',type=float,default=1)
parser.add_argument('--noEmptyPair', help='No 1-0 or 0-1 pairing',action="store_true",default=False)
parser.add_argument('--no2_2Group', help='No 2-2 pairing',action="store_true",default=False)
parser.add_argument('--penalty_n_n', help='Penalty score given for each n-n grouping',type=float,default=0.06)
parser.add_argument('--penalty_0_n', help='Penalty score given for each 0-n (or n-0) grouping (only used in lateGrouping)',type=float,default=0.15)


# other : persistance of embeddings
parser.add_argument('--useShelve',help='Save the embeddings in shelve (in order to quick up the next run)',action="store_true",default=False)
parser.add_argument('--skipLoadingModel',help='When using shelve loading labse may be skipped (only if shelve is complete)',action="store_true",default=False)

args = parser.parse_args()

# generic parameters
l1=args.l1
l2=args.l2
verbose=args.verbose
detectIntervals=args.detectIntervals
writeAlignableArea=args.writeAlignableArea
writeAnchorPoints=args.writeAnchorPoints
writeSegmentedInput=args.writeSegmentedInput
writeIntervals=args.writeIntervals
veryVerbose=args.veryVerbose
filePattern=re.compile(args.filePattern)
savePlot=args.savePlot
showPlot=args.showPlot
inputFile1=args.inputFile1
inputFile2=args.inputFile2
inputFileList=args.inputFileList
inputDir=args.inputDir
outputDir=args.outputDir
outputFilename=args.outputFilename
inputFormat=args.inputFormat # 'txt','arc','json'
xmlGuide=args.xmlGuide 
col1=args.col1
col2=args.col2
outputFormats=args.outputFormats
printIds=args.printIds
splitSent=args.splitSent
useSentenceSegmenter=args.useSentenceSegmenter
mergeLines=args.mergeLines
adaptativeMode=args.adaptativeMode

useNgrams=args.useNgrams
doNotRunDTW=args.doNotRunDTW
groupAfterOne2OnePairing=args.groupAfterOne2OnePairing
noMarginPenalty=args.noMarginPenalty
lateGrouping=args.lateGrouping

deltaDist=args.deltaDist
noEmptyPair=args.noEmptyPair
no2_2Group=args.no2_2Group
penalty_n_n=args.penalty_n_n
penalty_0_n=args.penalty_0_n
charRatio=args.charRatio

# sentence encoder method parameters
embedModel=args.embedModel
modelName=args.modelName
cosThreshold=args.cosThreshold
cosThresholdInOutputAnchors=args.cosThresholdInOutputAnchors
dtwBeam=args.dtwBeam
localBeamDecay=args.localBeamDecay
distNull=args.distNull

# ngram identification
n=args.ngram                # ngram size
diceThreshold=args.diceThreshold    # min dice to add a candidate point

# anchor point filtering parameters
deltaX=args.deltaX          # local space definition : +/-delta X on horizontal axis
deltaY=args.deltaY          # local space definition : +/-delta Y on vertical axis
minDensityRatio=args.minDensityRatio            # the minimal local density ratio (relatively to the average local density) to keep a candidate point
minHorizontalDensity=args.minHorizontalDensity  # the minimal density on horizontal axis to keep an interval in the final result
maxDistToTheDiagonal=args.maxDistToTheDiagonal  # the maximal distance to the diagonal (inside a given interval) for a point to be taken into account in the horizontal density
kBest=args.kBest            # number of best coordinates to keep in creating points
margin=args.margin          # margin : min distance between neighbouring sentences
maxGapSize=args.maxGapSize  # max distance between two points to make a gap between two interval
diagBeam=args.diagBeam      # max distance to the diagonal
localDiagBeam=args.localDiagBeam        # max distance to the diagonal in the interval
reiterateFiltering=args.reiterateFiltering
useShelve=args.useShelve
skipLoadingModel=args.skipLoadingModel
sentRatio=args.sentRatio


# various low level parameters
printLog=False
showPlot4NewInterval=False
minSentLengthRatio=0.2  # the minimal ratio between the shorter and the longer sentence to yield a candidate point
minSentLength=1         # the minimal sentence size to look for ngram
coeff_sent_len=0.33     # balance between sentence based length
coeff_neighbour_sim=0.6 # strength of the margin penalty
segMinLength=5 # min length for an aligned segment (in order to avoid oversegmentation)
useEncoder=False # to compute the embeddings of sentence concatenations
max_group_size=4
printGap=False


################################################################
# initialization code

infinite=float('inf')
allowed_groups=[]
printPlot=savePlot or showPlot
onlyOne2OnePairing=False

allowed_groups=[(0,1),(1,0),(1,1)]
if not onlyOne2OnePairing:
    for i in range(2,max_group_size+1):
        allowed_groups.append((1,i))
        allowed_groups.append((i,1))
    if noEmptyPair:
        allowed_groups.remove((1,0))
        allowed_groups.remove((0,1))
    if not no2_2Group:
        allowed_groups.append((2,2))
    
if verbose:
    print(f"Allowed groups : {allowed_groups}")

# to optimize parameters, temporarily save the embeddings in shelve or load embeds from the shelves
# N.B : embeds are normalized
if useShelve:
    embedShelve=shelve.open("embeds")
log=None

# opening log and models if necessary
if printLog:
    log=open(os.path.join(outputDir,"ailign.log"),mode="a",encoding="utf8")

# conditionnaly import alternative models (main model is labse)
preprocessor=False
encoder=False

# open various pretrained models (https://www.sbert.net/docs/pretrained_models.html) including labse
# n.b.: some model are more adapted to translation comparison, other to paraphrasing
if embedModel=="laser":
    # import modules for laser
    from laserembeddings import Laser
    encoder=Laser()
elif embedModel=="sbert":
    # import modules for sbert
    from sentence_transformers import SentenceTransformer
    print("*** Loading sbert model",)
    encoder = SentenceTransformer(modelName)
elif embedModel=="labse-keras":
    import tensorflow_hub as hub
    import tensorflow as tf
    import tensorflow_text as text  # Needed for loading universal-sentence-encoder-cmlm/multilingual-preprocess
    print("*** LABSE : Loading preprocessor")
    preprocessor = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
    print("*** LABSE : Loading model")
    encoder = hub.KerasLayer("https://tfhub.dev/google/LaBSE/2")


# init segmentation 
segmenter=None

# parameter for sentence segmentation
if splitSent:
    if useSentenceSegmenter:
        from trankit import Pipeline
        if verbose : 
            print("Loading sentence segmenter from trankit")
        # names are defined here : https://trankit.readthedocs.io/en/latest/pkgnames.html
        names={
            "en":"english",
            "de":"german",
            "es":"spanish",
            "fr":"french",
            "zh":"chinese",
            "ar":"arabic",
            "it":"italian"
        }
        try:
            segmenter = Pipeline(names[l1])
            segmenter.add(names[l2])
        except :
            print(f"Error while loading sentence segmenter from trankit. Check that you have defined a name for languages {l1} and {l2} (line 240)")
    else :
        # Rules that define sentence segmentation
        splitSent_regex={ 
            'zh': r'(?<=[：，。？！”])',
            'ar': r'(?<=\.|۔)',
            'fr': r'(?<=[.!?;:]) (?=[A-Z«"])|(?<=[!?;:])', # grimm Baudry
            'de': r'(?<=[.!?;:’“]) (?=[A-Z«"„])|(?<=[!?;:])|(?=[‘“])', # grimm KHM 1857
            'default' : r'(?<=[?;:.!"»…]) (?=[A-Z])',
        }

# Rules that define a correct end of line, for line merging


mergeLines_regex={ 
    'zh': r'[：，。？！”]\s*$',
    'fr': r'[?;:\.!"»…]\s*$',
    'ar': r'(\.|۔)\s*$'
}


########################################

# arc format is adapted to yasa input
arcHeader="\n<text>\n<divid='d1'>\n<pid='d1p1'>\n"
arcFooter="</p>\n</div>\n</text>\n"

# ces format is another standard for segmented files
cesHeader="""<?xml version="1.0" encoding="utf-8"?>
<cesAna>
<chunkList>
<chunk>
<par>
"""
cesFooter="""
</par>
</chunk>
</chunkList>
</cesAna>"""

# cesalign format is used to store alignment result
cesAlignHeader=f"""<?xml version="1.0" encoding="utf-8"?>

<cesAlign type="seg" version="1.6">

<cesHeader version="2.3" meanScore="__meanScore__">
    <translations>
        <translation lang="{l1}" />
        <translation lang="{l2}" />
    </translations>
</cesHeader>

<linkList>
    <linkGrp targType="seg">
    
"""
cesAlignFooter="""
</linkGrp>
</linkList>

</cesAlign>
"""
# tmx is a common xml format to encode aligned file (for translation memories)
tmxHeader="""
<?xml version="1.0" encoding="utf-8" ?>
<!DOCTYPE tmx SYSTEM "tmx14.dtd">
<tmx version="1.4">
  <header
    creationtool="AIlign"
    creationtoolversion="1.0"
    datatype="unknown"
    segtype="sentence"
    meanScore="__meanScore__"
  >
  </header>
  <body>
"""

tmxFooter="""
  </body>
</tmx>  
"""

def toXML (s):
    s=re.sub(r'&','&amp;',s);
    s=re.sub(r'<','&lt;',s);
    s=re.sub(r'>','&gt;',s);
    return s
    

########################################################################### filtering points functions
# computation of local density 
# local space may be centered, or before (for a point wich ends an interval) 
# or after (for a point that begins an interval).
# max density is taken
def computeLocalDensity(i,j,points,I,J,simMat,deltaX,deltaY):
    coeff=J/I if sentRatio==0 else sentRatio
    localSpaceSizeBefore=0
    nbPointsInLocalSpaceBefore=0

    localSpaceSizeCentered=0
    nbPointsInLocalSpaceCentered=0
    
    localSpaceSizeAfter=0
    nbPointsInLocalSpaceAfter=0
    
    for X in range(max(0,i-2*deltaX),min(i+2*deltaX+1,I)):
        for Y in range(int(max(0,j-(i-X)*coeff-deltaY)),int(min(j-(i-X)*coeff+deltaY+1,J))):
            if X <= i:
                localSpaceSizeBefore+=1
                if (X,Y) in points.keys():
                    nbPointsInLocalSpaceBefore+=simMat[X,Y]
            if X >= i:
                localSpaceSizeAfter+=1
                if (X,Y) in points.keys():
                    nbPointsInLocalSpaceAfter+=simMat[X,Y]
            if max(0,i-deltaX) <= X < min(i+deltaX+1,I):
                localSpaceSizeCentered+=1
                if (X,Y) in points.keys():
                    nbPointsInLocalSpaceCentered+=simMat[X,Y]
                    
    (densityBefore,densityAfter,densityCentered)=(0,0,0)
    if localSpaceSizeBefore:
        densityBefore=nbPointsInLocalSpaceBefore/localSpaceSizeBefore
    if localSpaceSizeAfter:
        densityAfter=nbPointsInLocalSpaceAfter/localSpaceSizeAfter
    if localSpaceSizeCentered:
        densityCentered=nbPointsInLocalSpaceCentered/localSpaceSizeCentered
    return max(densityBefore,densityAfter,densityCentered)


# filtering points by eliminating every point in the center of a low density local area
# output : (points,filtered_x,filtered_y)
def filterPoints(points,I,J,averageDensity,simMat,deltaX,deltaY):
    # initialisation of filtered points
    filtered_x=[]
    filtered_y=[]
    nbDeleted=0
    
    if veryVerbose:
        print("Filtering ",len(points),"...")

    # computation of local density for each point
    pointsKey=sorted(list(points.keys()),key=lambda point:point[0])
    
    for point in pointsKey:
        (i,j)=point
        
        localDensity=computeLocalDensity(i,j,points,I,J,simMat,deltaX,deltaY)
        
        if veryVerbose:
            print ("i=",i,"j=",j,"Local density=",localDensity,"Average density=",averageDensity,"Ratio=",round(localDensity/averageDensity,2))
        
        # point is removed if density is not high enough
        if averageDensity>0 and localDensity/averageDensity < minDensityRatio:
            del(points[(i,j)])
            nbDeleted+=1

            #~ x=[p[0] for p in points ]
            #~ y=[p[1] for p in points]
            #~ plt.axis([0,I,0,J])
            #~ plt.title(str(i)+","+str(j)+'=> low density')
            #~ plt.scatter(x,y,c="black",s=1) 
            #~ plt.scatter([i],[j],c="red",s=1)                      
            #~ (i1,j1)=(i-deltaX,j-deltaX-deltaY)
            #~ (i1,j2)=(i-deltaX,j-deltaX+deltaY)
            #~ (i2,j3)=(i+deltaX,j+deltaX+deltaY)
            #~ (i2,j4)=(i+deltaX,j+deltaX-deltaY)
            #~ X=[i1,i1,i2,i2,i1]
            #~ Y=[j1,j2,j3,j4,j1]
            #~ plt.plot(X,Y,c="grey")
            #~ plt.show()
            
        else:
            filtered_x.append(i)
            filtered_y.append(j)
    
    if verbose:
        print(nbDeleted,"points have been removed!")
    
    return (points,filtered_x,filtered_y)

# removing points that are conflicting on the same column : only the point with the higher local density is kept
def resolvingConflicts(points,I,J,simMat):
    x2y={}
    y2x={}
    filtered_x=[]
    filtered_y=[]
    nbDeleted=0
    pointsKey=list(points.keys())
    for point in pointsKey:
        (i,j)=point
        # conflict on x coordinate
        if i in x2y.keys():
            if x2y[i]!=j:
                # for x coordinate, conflict between (i,j) and (i,x2y[i])
                # only the best point is kept
                density1=computeLocalDensity(i,j,points,I,J,simMat,deltaX,deltaY)
                density2=computeLocalDensity(i,x2y[i],points,I,J,simMat,deltaX,deltaY)
                nbDeleted+=1
                if density1 > density2:
                    if (i,x2y[i]) in points:
                        del(points[(i,x2y[i])])
                    x2y[i]=j
                else:
                    del(points[(i,j)])
                    continue
        else:
            x2y[i]=j
            
        if j in y2x.keys():
            if y2x[j]!=i:
                # for x coordinate, conflict between (i,j) and (i,x2y[i])
                # only the best point is kept
                density1=computeLocalDensity(i,j,points,I,J,simMat,deltaX,deltaY)
                density2=computeLocalDensity(y2x[j],j,points,I,J,simMat,deltaX,deltaY)
                nbDeleted+=1 
                if density1 < density2:
                    if (y2x[j],j) in points:
                        del(points[(y2x[j],j)])
                    y2x[j]=i
                else:
                    del(points[(i,j)])
        else :
            y2x[j]=i

    if verbose:
        print(nbDeleted,"conflicting points have been removed!")
    
    pointsKey=list(points.keys())
    for point in pointsKey:
        (i,j)=point
        filtered_x.append(i)
        filtered_y.append(j)
    return (points,filtered_x,filtered_y)

########################################################################### ngram points functions

# ngram that contain only the same repeated character are not valid (e.g. blank spaces...)
def valid(ngram):
    return not re.match(r'(.)\1+',ngram)

# extract candidates points using ngram search
def computePointsFromNgrams(sents1,sents2):
    # extracting hash table that records all the ngrams for sents1
    lenSents1=len(sents1)
    lenSents2=len(sents2)
    
    ngrams1=[]
    for i in range(lenSents1):
        ngrams1.append({})
        sent1=sents1[i]
        for k in range(0,len(sent1)-n):
            ngram=sent1[k:k+n]
            if valid(ngram):
                if ngram not in ngrams1[i].keys() :
                    ngrams1[i][ngram]=0
                ngrams1[i][ngram]+=1

    # extracting hash table that records all the ngrams for sents2
    ngrams2=[]
    for j in range(lenSents2):
        sent2=sents2[j]
        ngrams2.append({})
        for k in range(0,len(sent2)-n):
            ngram=sent2[k:k+n]
            if valid(ngram):
                if ngram not in ngrams2[j].keys():
                    ngrams2[j][ngram]=0
                ngrams2[j][ngram]+=1
    
    # record the corresponding coordinate, sorted according to dice
    bestJ={}
    bestI={}

    # Using diagBeam param
    if diagBeam:
        range2=lenSents2*diagBeam
    else : 
        range2=lenSents2
    # dice computation for each point (i,j)
    for i in range(lenSents1):
        nb1=max(1,len(sents1[i])-n+1)
        if verbose and i%100==0:
            print ("x =",i,"/",lenSents1)
        for J in range(range2):
            if diagBeam:
                # when using fixed vertical width around diag, j must be computed as: int(i*lenSents2/lenSents1-range2/2)
                j=int(i*lenSents2/lenSents1-range2/2)
            else:
                j=J
            if j<0:
                continue
            nb2=max(1,len(sents2[j])-n+1)
            # length of sent1 and sent2 must be comparable
            if nb1>minSentLength and nb2>minSentLength and nb1/nb2 >= minSentLengthRatio and nb2/nb1 >=minSentLengthRatio:
                # computing the number of common ngrams (based on occurrences and not on type)
                nbCommon=0
                for ngram in ngrams1[i].keys():
                    if ngram in ngrams2[j].keys():
                        nbCommon+=min(ngrams1[i][ngram],ngrams2[j][ngram])
                dice=2*nbCommon/(nb1+nb2)
                # if dice is greater than the threshold, candidate point (i,j) is recorded
                if dice>diceThreshold:
                    if not j in bestI.keys():
                        bestI[j]=[]
                    if not i in bestJ.keys():
                        bestJ[i]=[]
                    bestI[j].append((dice,i))
                    bestJ[i].append((dice,j))
    return kBestPoints(bestI,bestJ)

def kBestPoints(bestI,bestJ):
    # building the point list taking, for each coordinate, the k best corresponding point
    x=[]
    y=[]
    points={} # points are recorded here as keys
    for i in bestJ.keys():
        # sorting the candidate according to sim
        bestJ[i]=sorted(bestJ[i],key = lambda x:x[0],reverse=True)
        if len(bestJ[i])>1:
            if (bestJ[i][0][0]-bestJ[i][1][0]) < margin:
                if verbose:
                    print("Filtering using margin criterion : ",bestJ[i][0][0],"-",bestJ[i][1][0],"<",margin)
                bestJ[i]=()
            else:
                # only the k best are recorded
                bestJ[i]=[bestJ[i][l][1] for l in range(0,min(kBest,len(bestJ[i])))]
        
    for j in bestI.keys():
        # sorting the candidate according to dice
        bestI[j]=sorted(bestI[j],key = lambda x:x[0],reverse=True)
        if len(bestI[j])>1:
            if (bestI[j][0][0]-bestI[j][1][0]) < margin:
                if verbose:
                    print("Filtering using margin criterion : ",bestI[j][0][0],"-",bestI[j][1][0],"<",margin)
                bestI[j]=()
            else:   
                # only the k best are recorded
                bestI[j]=[bestI[j][l][1] for l in range(0,min(kBest,len(bestI[j])))]
    
    for i in bestJ.keys():  
        for j in bestJ[i]:
            if j in bestI and i in bestI[j]:
                x.append(i)
                y.append(j)
                points[(i,j)]=1
    return (points,x,y)

############################################################# LABSE points functions
# Function to normalize the embeddings by dividing them with their L2-norm
def normalization(embeds):
    norms = np.linalg.norm(embeds, 2, axis=1, keepdims=True)
    return embeds / norms
    
def computePointsWithEncoder(preprocessor,encoder,sents1,sents2):
    points={} # points are recorded here as keys
    
    t0=time.time()
    
    runEncoder=True
    # load from shelve in test mode (embeds are already computed)
    if useShelve:
        embeds1=[]
        embeds2=[]
        runEncoder=False
        for sent in sents1:
            if sent in embedShelve:
                embeds1.append(embedShelve[sent])
            else:
                runEncoder=True
                break
        for sent in sents2:
            if sent in embedShelve:
                embeds2.append(embedShelve[sent])
            else:
                runEncoder=True
                break
    if runEncoder:
        if verbose:
            print("Running Encoder...\n")
            
        embeds1 = computeEmbeds(preprocessor,encoder,embedModel,sents1,l1)
        embeds2 = computeEmbeds(preprocessor,encoder,embedModel,sents2,l2)

        t1=time.time()
        if verbose:
            print("\n1. Encoding -->",t1-t0,"s.\n")

        # For semantic similarity tasks, apply l2 normalization to embeddings
        embeds1 = normalization(embeds1)
        embeds2 = normalization(embeds2)
        t2=time.time()
        if verbose:
            print("\n2. Normalization -->",t2-t1,"s.\n")
    
    # saving normalized embeddings to shelve
    if useShelve and runEncoder:
        for i,sent in enumerate(sents1):
            embedShelve[sent]=embeds1[i]
        for i,sent in enumerate(sents2):
            embedShelve[sent]=embeds2[i]
        t2=time.time()
        if verbose:
            print("1-2. Loading embeddings from shelve -->",t2-t0,"s.\n"),
    t3=time.time()
    # similarity
    mat=np.matmul(embeds1, np.transpose(embeds2))

    t4=time.time()
    if verbose:
        print("\n3. Similarity matrix -->",t4-t3,"s.\n"),
    
    # building the point list taking, for each coordinate, the k best corresponding point
    x=[]
    y=[]
    points={} # points are recorded here as keys

    # if the searchspace is reduced around the diagonal, compute the kBest point manually
    if diagBeam<1:
        # record the corresponding coordinate
        bestJ={}
        bestI={}
        maxVertDistToTheDiagonal=int(len(sents2)*diagBeam)
        for i in range(len(mat)):
            diagJ=int(i/len(mat)*len(mat[i]))
            infJ=max(0,diagJ-maxVertDistToTheDiagonal)
            supJ=min(diagJ+maxVertDistToTheDiagonal,len(mat[i]))
            for j in range(infJ,supJ):
                if mat[i][j]>cosThreshold:
                    if i not in bestJ:
                        bestJ[i]=[]
                    if j not in bestI:
                        bestI[j]=[] 
                    bestJ[i].append((mat[i][j],j))
                    bestI[j].append((mat[i][j],i))
        t5=time.time()
        if verbose:
            print("\n4. Extracting points -->",t5-t4,"s.\n"),

        (points,x,y)= kBestPoints(bestI,bestJ)      
        t6=time.time()
        if verbose:
            print("\n5. Filtering k best vertically and horizontally -->",t6-t5,"s.\n"),

    # use numpy argpartition for kBest extraction
    else:
        k=kBest
        # for k=1 we extract the 2 best, in order to apply the margin criterion
        if k==1:
            k=2
        k=min(k,len(mat[0]))
        # using argpartition allow to extract quickly the k-best col for each line
        ind_by_line = np.argpartition(mat,-k,axis=1)[:,-k:]
        sim_by_line = np.take_along_axis(mat, ind_by_line, axis=1)
        bestJ=[list(zip(sim_by_line[i],ind_by_line[i])) for i in range(len(sim_by_line))]

        for i in range(len(bestJ)):
            bestJ[i].sort(key=lambda x:x[0],reverse=True)
            if (bestJ[i][0][0]-bestJ[i][1][0]) < margin:
                if veryVerbose:
                    print("Filtering using margin criterion : ",bestJ[i][0][0],"-",bestJ[i][1][0],"<",margin)
                bestJ[i]=[]
            # once margin criterion has been applied, apply the threshold
            bestJ[i]=[pair for pair in bestJ[i] if pair[0]>cosThreshold]
            # if kBest==1, crop the candidate list
            if kBest==1 and len(bestJ[i])>1:
                bestJ[i]=bestJ[i][0:1]
        
        ind_by_col = np.argpartition(mat,-k,axis=0)[-k:,:]
        sim_by_col = np.take_along_axis(mat, ind_by_col, axis=0)
        ind_by_col = ind_by_col.swapaxes(1,0)
        sim_by_col = sim_by_col.swapaxes(1,0)
        
        bestI=[list(zip(sim_by_col[i],ind_by_col[i])) for i in range(len(ind_by_col))]
        for j in range(len(bestI)):
            bestI[j].sort(key=lambda x:x[0],reverse=True)
            if (bestI[j][0][0]-bestI[j][1][0]) < margin:
                if veryVerbose:
                    print("Filtering using margin criterion : ",bestI[j][0][0],"-",bestI[j][1][0],"<",margin)
                bestI[j]=[]
            # once margin criterion has been applied, apply the threshold               
            bestI[j]=[pair for pair in bestI[j] if pair[0]>cosThreshold]
            # if kBest==1, crop the candidate list
            if kBest==1 and len(bestI[j])>1:
                bestI[j]=bestI[j][0:1]

        # adding points
        for i in range(len(bestJ)):
            for n in range(len(bestJ[i])):
                j=bestJ[i][n][1]
                for m in range(len(bestI[j])):
                    if i==bestI[j][m][1]:
                        x.append(i)
                        y.append(j)
                        points[(i,j)]=1
                        break

        t5=time.time()
        if verbose:
            print("\n4-5. Extracting and filtering k best vertically and horizontally -->",t5-t4,"s.\n"),


    return (points,x,y,mat,embeds1,embeds2)

############################################################# LASER and BERT points functions 



# return the normalized embeddings for a given encoder and a sentence list
def computeEmbeds(preprocessor,encoder,embedModel,sents,language=""):
    if embedModel == "laser":
        # Use the Laser model to embed the sentences in different languages
        embeds = encoder.embed_sentences(sents, language)
        
    else:
        if preprocessor:
            embeds = encoder(preprocessor(sents))["default"]
        else:
            embeds = encoder.encode(sents)

    # Normalize the embeddings using the normalization function
    embeds = normalization(embeds)
    return embeds
    

# Function to compute the similarity matrix and identify similar sentences
def computePoints(tokenizer, model, embedModel, sents1, sents2):
    points = {}  # Dictionary to store the indices of similar sentences
    t0 = time.time()  # Record the starting time for performance measurement
    
    if embedModel == "bert":
        if verbose:
            print("Running Encoder...\n"),
        # Tokenize the input sentences using the BERT tokenizer
        inputs1 = tokenizer(sents1, return_tensors='pt', padding=True, truncation=True)
        inputs2 = tokenizer(sents2, return_tensors='pt', padding=True, truncation=True)
        t1 = time.time()
        if verbose:
            print("1. Encoding -->", t1 - t0, "s.\n")  # Print the time taken for encoding
        
        # Pass the tokenized inputs through the BERT model
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
        
        # Extract the embeddings from the BERT model's output and convert them to numpy arrays
        embeds1 = outputs1.last_hidden_state[:, 0, :]
        embeds2 = outputs2.last_hidden_state[:, 0, :]
        
        # Normalize the embeddings using the normalization function
        embeds1 = normalization(embeds1.detach().numpy())
        embeds2 = normalization(embeds2.detach().numpy())
        t2 = time.time()
        if verbose:
            print("2. Normalization-->", t2 - t1, "s.\n")  # Print the time taken for normalization
        
    elif embedModel == "laser":
        # Use the Laser model to embed the sentences in different languages
        embeds1 = laser.embed_sentences(sents1, lang='fr')
        embeds2 = laser.embed_sentences(sents2, lang='en')
        t1 = time.time()
        if verbose:
            print("1. Encoding -->", t1 - t0, "s.\n")  # Print the time taken for encoding
        
        # Normalize the embeddings using the normalization function
        embeds1 = normalization(embeds1)
        embeds2 = normalization(embeds2)
        t2 = time.time()
        if verbose:
            print("2. Normalization-->", t2 - t1, "s.\n")  # Print the time taken for normalization
    
    # Compute the similarity matrix between the embeddings of the two sets of sentences
    mat = np.matmul(embeds1, embeds2.T)
    t3 = time.time()
    if verbose:
        print("3. Similarity matrix -->", t3 - t2, "s.\n")  # Print the time taken for computing the similarity matrix
    
    x = []  # List to store the indices of similar sentences from sents1
    y = []  # List to store the indices of similar sentences from sents2
    bestJ = {}  # Dictionary to store the best match index for each sentence in sents1
    bestI = {}  # Dictionary to store the best match index for each sentence in sents2
    
    # Find the best match index for each sentence in sents1
    for i in range(len(mat)):
        m = 0
        for j in range(len(mat[i])):
            if mat[i][j] > m:
                m = mat[i][j]
                bestJ[i] = j
    
    # Find the best match index for each sentence in sents2
    for j in range(len(mat[0])):
        m = 0
        for i in range(len(mat)):
            if mat[i][j] > m:
                m = mat[i][j]
                bestI[j] = i
    t4 = time.time()
    if verbose:
        print("4. Extracting best point according to horizontal and vertical axis-->", t4 - t3, "s.\n")  # Print the time taken for computing the similarity matrix

    
    # Identify the similar sentence pairs based on the best match indices and similarity threshold
    for i in range(len(mat)):
        j = bestJ[i]
        if bestI.get(j) == i and mat[i][j] >= cosThreshold:
            x.append(i)
            y.append(j)
            points[(i, j)] = 1  # Store the indices of similar sentences in the points dictionary
    t5 = time.time()
    if verbose:
        print("5. Filtering best points that exceed the threshold -->", t5 - t4, "s.\n")  # Print the time taken for computing the similarity matrix
    
    
    return points, x, y, mat, embeds1, embeds2

######################################################################### reading / writing files
# reading input file

def readInputFile(inputDir,inputFile,inputFormat,column=0,language="fr"):
    """Reads an input file and returns a list of sentences.

      Args:
        inputDir: The directory containing the input file.
        inputFile: The name of the input file.
        inputFormat: The format of the input file.
        column: The column number of the input file that contains the text.
        language: The language of the input file.

      Returns:
        sents: a list of sentences.
        idSents: the list of sentence ids (build upon segment ids)
        lenSents: the number of sentences
        seg2sents: a list of list of integer, that gives the 1-n correspondence
            between an original segment number and the list of final sentences
            - if splitSent, for one segment, we may have more than one sentences
            - if mergeSent, more than one segment may correspond to the same merged sentence
    """
    global segMinLength
    
    segs=[]
    idSegs=[]
    lenSents=0
    seg2sents=[]
    nbChars=0

    try:
        inputFilePath=os.path.join(inputDir, inputFile) if inputDir else inputFile
        f = open(inputFilePath, encoding='utf8')
    except:
        print("Error: a problem occurred while opening", inputFile)
        sys.exit()

    # Reading according to inputFormat
    if inputFormat == "arc" or inputFormat == "ces":
        for line in f:
            line = line.strip()
            m = re.search(r'<s\b[^>]+id="([^"]*)"', line)
            if m and m.group(1):
                idSegs.append(m.group(1))
            else:
                idSegs.append("s"+str(lenSegs))
            segs.append(line)
            lenSegs += 1
            nbChars +=len(line)


    # The json input contains a sentences property, which is a list sentences, which are list of tokens
    # Each token is a list of conll features, col1->form, col9=blank_space
    elif inputFormat == "json":
        content = f.read()
        jsonObj = json.loads(content)
        segs = [
            "".join([tok[1] + tok[9] for tok in sent if len(tok)>=10]) for sent in jsonObj["sentences"]
        ]
        for seg in segs :
            nbChars+=len(seg)
        idSegs=[str(i) for i in list(range(0,len(segs)+1))]
        
    # the tsv format is an already aligned format. Sentence are extracted from a specific column
    elif inputFormat == "tsv":
        segs = []
        for line in f:
            alignedSegs=re.split("\t",line)
            segs.append(alignedSegs[column])
            nbChars +=len(alignedSegs[column])
        idSegs=[str(i) for i in list(range(1,len(segs)+1))]

    # in xml-conll, the conll sentences are encoded between <s></s> markup
    elif inputFormat == "xml-conll":
        content = f.read()
        try:
            xmlRoot = ET.fromstring(content)
        except:
            print("non conform XML :",os.path.join(inputDir, inputFile))
            # error_log.write("non conform XML :",os.path.join(inputDir, inputFile),"\n")
            sys.exit() 

        for sElt in xmlRoot.findall('.//s'):
            s="".join(sElt.itertext())
            toks=[]
            for line in re.split(r"\n",s):
                cols=re.split("\t",line)
                if len(cols)==10:
                    toks.append(cols[1])
            seg=" ".join(toks)
            segs.append(seg)
            nbChars+=len(seg)
            
            if sElt.attrib["id"]:
                idSegs.append(sElt.attrib["id"])
            elif sElt.attrib["xml-id"]:
                idSegs.append(sElt.attrib["xml-id"])
            else:
                idSegs.append(str(len(segs)))
    
    # In XML format, the sentences are extracted using the text content of
    # the elements that are defined by xmlGuide (a list of tag or simple xpath expressions)
    elif inputFormat == "xml":
        content = f.read()
        try:
            xmlRoot = ET.fromstring(content)
        except:
            print("non conform XML :",os.path.join(inputDir, inputFile))
            # error_log.write("non conform XML :",os.path.join(inputDir, inputFile),"\n")
            sys.exit() 
        segs = []
        xpath='|'.join(['//'+tag for tag in xmlGuide])
        for elt in xmlRoot.findall(xpath):
            content="".join(elt.itertext())
            segs.append(content)
            nbChars+=len(content)
            # recording id in idSegs
            if elt.attrib["id"]:
                idSegs.append(elt.attrib["id"])
            elif elt.attrib["xml-id"]:
                idSegs.append(elt.attrib["xml-id"])
            else:
                idSegs.append(str(len(segs)))

    # Default format: one sentence per line
    else:
        print ("Warning : default format TXT")
        for line in f:
            line=line.strip()
            line=re.sub(r'\x0A|\x0D','',line)
            nbChars+=len(line)
            segs.append(line)
        idSegs=[str(i) for i in list(range(1,len(segs)+1))]

    # Here, the lines that corresponds to the same sentences may be merged
    # The corresponding sentence id will result in the concatenation of initial segment id
    if mergeLines:
        if verbose: 
            print("Line merging for ",language)
        sents=[]
        idSents=[]
        numSents=0
        currentSent=[]
        currentIds=[]
        for (i,seg) in enumerate(segs):
            seg2sents.append([numSents])
            currentIds.append(idSegs[i])
            currentSent.append(seg)
            # merging when the sentence ends with a separator
            if re.search(mergeLines_regex[language],seg) or seg.upper()== seg:
                idSents.append("-".join(currentIds))
                sents.append(" ".join(currentSent))
                currentSent=[]
                currentIds=[]
                numSent+=1
        if len(currentIds)>0:
            sents.append(" ".join(currentSent))
            idSents.append("-".join(currentIds))

    # here, segments can be split in smaller pieces
    elif splitSent:
        if verbose: 
            print("Sentence segmentation for ",language)
        sents=[]
        idSents=[]
        if useSentenceSegmenter:
            segmenter.set_active(names[language])
        for (i,seg) in enumerate(segs):
            # use trankit for sentence segmentation
            if useSentenceSegmenter:
                print("segmentation de ",seg)
                sentences=segmenter.ssplit(seg)['sentences']
                someSents = [sent['text'] for sent in sentences]
            # or use a set of regex declared in splitSent
            else:
                if language in splitSent_regex:
                    regex=splitSent_regex[language]
                else:
                    regex=splitSent_regex["default"]
                someSents=re.split(splitSent_regex[language],seg)
                
            lastSent=""
            newSents=[]
            # the splitted segment that are too small (< segMinLength)
            # are grouped with the follower
            for sent in someSents:
                if len(lastSent+sent) > segMinLength:
                    newSents.append(lastSent+" "+sent)
                    lastSent=""
                else:
                    if lastSent=="":
                        lastSent=sent
                    else:
                        lastSent+=" "+sent
            if lastSent:
                newSents.append(lastSent)
                
            seg2sents.append(list(range(len(sents),len(sents)+len(newSents))))
            newIds=[idSegs[i]]
            if len(newSents)>1:
                newIds=[ idSegs[i]+"_"+str(j) for j in range(len(newSents)) ]
            sents.extend(newSents)
            idSents.extend(newIds)
        
    # keeping the same segments as in the input
    else:
        sents=segs
        idSents=idSegs
        seg2sents=[ [j] for j in range(len(sents)) ]

    lenSents=len(sents)
    if verbose: 
        print(lenSents,"sentences for ",language)
        if veryVerbose:
            print("\n".join(sents))
            
    f.close()
    
    if writeSegmentedInput:
        inputFilePathSeg=re.sub(r"(.*)([.]\w+)[.]\w+$",r"\1.seg\2.txt",inputFilePath)
        segFile=open(inputFilePathSeg,mode="w",encoding="utf8")
        segFile.write("\n".join(sents))
        segFile.close()
    
    return (sents,idSents,lenSents,seg2sents,nbChars)

# write only alignable intervals of l1 or l2 file
def writeAlignable(sents,idSents,intervals,index,outputDir,outputFile,outputFormat):
    """
    Arguments :
        sents : List(str) : the sentence list
        idSents : List(str) : the corresponding sentence ids    
        intervals : List(List(int)) : the list of pairs [i..j] that defines corresponding intervals. The second axe is the language : 0 for l1, 1 for l2
        index : 0 or 1 for l1 or l2
        outputDir : str : the path of output dir
        outputFile : str : the name of output file
        outputFormat : str : "ces" or "arc" or "txt"
        
    No return value, but the file outputFile is written on the disk
    """
    
    try:
        output=open(os.path.join(outputDir,outputFile),mode="w",encoding="utf8")
        # output header
        if outputFormat=="ces":
            output.write(cesHeader)
        elif outputFormat=="arc":
            output.write(arcHeader)

        # output sentences
        for interval in intervals:
            i1=interval[0][index]
            i2=interval[1][index]
            
            for i in range(i1,i2+1):
                if outputFormat=="ces" or outputFormat=="arc":
                    if inputFormat=="ces" or outputFormat=="arc":
                        idSent=idSents[i]
                    else:
                        idSent=str(i+1)
                    output.write("<s id=\""+idSent+"\">\n"+toXML(sents[i])+"\n</s>\n")
                else :
                    output.write(sents[i]+"\n")

        # output footer
        if outputFormat=="ces":
            output.write(cesFooter)
        elif outputFormat=="arc":
            output.write(cesFooter)

        output.close()
    except:
        print ("Error: a problem occurred while writing",outputFile)
        sys.exit()

# write aligned points
# if the anchor parameter is true then filtered_x and filtered_y are list of int
# if not, they are list of list of int (the grouped coordinate)
# NB : fonction d'écriture des sorties à compléter par INES - 
# TODO : option printIds
def writeAlignedPoints(l1,l2,sents1,idSents1,sents2,idSents2,filtered_x,filtered_y,outputDir,outputFile,outputFormat,anchor,printIds=False,meanScore=0):
    """
    Arguments :
        sents1 : List(str) : the L1 sentence list
        idSents1 : List(str) : the corresponding sentence ids   
        sents2 : List(str) : the L2 sentence list
        idSents2 : List(str) : the corresponding sentence ids   
        filtered_x : List(List(int)) OR List(int) if anchor=True 
            if anchor == false : the X coordinates of groups in L1 (ex. :[[0],[1,2],[3],[4,5,6]])
            if anchor == true : the X coordinates of points in L1 (ex. [0, 2, 3, 5])
        filtered_y : List(List(int)) OR List(int) if anchor=True
            if anchor == false : the Y coordinates of groups in L2 (ex. :[[0,1],[2],[3],[4,5]])
            if anchor == true : the Y coordinates of points in L2 (ex. [1, 2, 3, 5])
        outputDir : str : the path of output dir
        outputFile : str : the name of output file
        outputFormat : str : "tmx" or "ces" or "ids" or "txt"
    
    No return value, but the file outputFile is written on the disk
    """
    
    global tmxHeader,cesAlignHeader

    #~ try:
    if outputFormat == "txt2":
        outputFile1=outputFile.replace(".txt2","."+l1+".txt")
        output1=open(os.path.join(outputDir,outputFile1),mode="w",encoding="utf8")
        outputFile2=outputFile.replace(".txt2","."+l2+".txt")
        output2=open(os.path.join(outputDir,outputFile2),mode="w",encoding="utf8")        
    else:
        output=open(os.path.join(outputDir,outputFile),mode="w",encoding="utf8")
    #~ output2=open(os.path.join(outputDir,outputFile+".txt"),mode="w",encoding="utf8")
    # output header
    if outputFormat=="ces":
        output.write(re.sub(r'__meanScore__',f"{meanScore:.4f}",cesAlignHeader))
    elif outputFormat=="tmx":
        output.write(re.sub(r'__meanScore__',f"{meanScore:.4f}",tmxHeader))
    elif outputFormat=="txt":
        output.write(f"Mean similarity:{meanScore}\n")
    elif outputFormat=="txt2":
        output1.write(f"Mean similarity:{meanScore}\n")
        output2.write(f"Mean similarity:{meanScore}\n")

    # output sentences
    for i in range(len(filtered_x)):
        if anchor:
            x=[filtered_x[i]]
            y=[filtered_y[i]]
        else:
            x=filtered_x[i]
            y=filtered_y[i]
        if outputFormat=="ces":
            if inputFormat=="ces" or inputFormat=="arc":
                idSent1=" ".join([idSents1[x[j]] for j in range(len(x))])
                idSent2=" ".join([idSents2[y[j]] for j in range(len(y))])
            else:
                idSent1=" ".join([str(x[j]+1) for j in range(len(x))])
                idSent2=" ".join([str(y[j]+1) for j in range(len(y))])
            output.write(f"\t\t<link xtargets=\"{idSent1} ; {idSent2}\"/>\n")
        elif outputFormat=="ids":
            if inputFormat=="ces" or inputFormat=="arc":
                idSent1=" ".join([idSents1[x[j]] for j in range(len(x))])
                idSent2=" ".join([idSents2[y[j]] for j in range(len(y))])
            else:
                idSent1=" ".join([str(x[j]+1) for j in range(len(x))])
                idSent2=" ".join([str(y[j]+1) for j in range(len(y))])
                output.write(f"{idSent1}\t{idSent2}\n")
        elif outputFormat == "tmx":
            srcSegs = "".join([ "\t\t<seg>"+toXML(sents1[x[j]])+"</seg>\n" for j in range(len(x)) ])
            tgtSegs = "".join([ "\t\t<seg>"+toXML(sents2[y[j]])+"</seg>\n" for j in range(len(y)) ])

            output.write(f"<tu>\n")
            output.write(f"\t<tuv xml:lang=\"{l1}\">\n{srcSegs}\t</tuv>\n")
            output.write(f"\t<tuv xml:lang=\"{l2}\">\n{tgtSegs}\t</tuv>\n")
            output.write(f"</tu>\n")

        elif outputFormat == "txt":
            ids1 = "[" + " ".join([str(x[j]) for j in range(len(x))]) + "] " if printIds else ""
            sent1 = ids1 + " ".join([sents1[x[j]] for j in range(len(x))])
            ids2 =  "[" + " ".join([str(y[j]) for j in range(len(y))]) + "] " if printIds else ""
            sent2 = ids2 + " ".join([sents2[y[j]] for j in range(len(y))])
            output.write(sent1 + "\n" + sent2 + "\n\n")
        elif outputFormat == "txt2":
            sent1 = " ".join(["["+str(x[j])+"] "+sents1[x[j]] for j in range(len(x))])
            output1.write(sent1 + "\n")
            sent2 = " ".join(["["+str(y[j])+"] "+sents2[y[j]] for j in range(len(y))])
            output2.write(sent2 + "\n")
        elif outputFormat == "tsv":
            ids1 = "[" + " ".join([str(x[j]) for j in range(len(x))]) + "] " if printIds else ""
            sent1 = " ".join([sents1[x[j]] for j in range(len(x))])
            ids2 = "[" + " ".join([str(y[j]) for j in range(len(y))]) + "] " if printIds else ""
            sent2 = " ".join([sents2[y[j]] for j in range(len(y))])
            output.write(f"{ids1}{sent1}\t{ids2}{sent2}\n")
        elif outputFormat == "bertalign":
            ids1 = "[" + ",".join([str(x[j]) for j in range(len(x))]) + "]"
            ids2 = "[" + ",".join([str(y[j]) for j in range(len(y))]) + "]"
            output.write(f"{ids1}:{ids2}\n")
        else :
            ids1=  "["+" ".join([ str(x[j]) for j in range(len(x))])+"] " if printIds else ""
            sent1=ids1+" ".join([ sents1[x[j]] for j in range(len(x))])
            ids2=  "["+" ".join([ str(y[j]) for j in range(len(y))])+"] " if printIds else ""
            sent2=ids2+" ".join([ sents2[y[j]] for j in range(len(y))])
            output.write(sent1+"\t"+sent2+"\n")

    # output footer
    if outputFormat=="ces":
        output.write(cesAlignFooter)
    elif outputFormat=="tmx":
        output.write(tmxFooter)
    
    if outputFormat == "txt2":
        output1.close()
        output2.close()
    else:
        output.close()


def extractAnchorPoints(points,x,y,sents1,sents2,lenSents1,lenSents2,simMat):
    anchor_points=dict.copy(points)
    
    # =====> STEP 6 : compute average local density around selected points
    t5=time.time()
   
    pointsKey=list(anchor_points.keys())

    if len(pointsKey)==0:
        print("No anchor points !!!")
        beginInt=(-1,-1)
        lastI=lenSents1-1
        lastJ=lenSents2-1
        intervalLengthSent1+=lastI - beginInt[0] + 1
        intervalLengthSent2+=lastJ - beginInt[1] + 1
        for n in range(0,lastI+1):
            intervalLengthChar1+=len(sents1[n])
        for n in range(0,lastJ+1):
            intervalLengthChar2+=len(sents2[n])
        
    else:
        totDensity=0
        for point in pointsKey:
            (x2,y2)=point
            totDensity+= computeLocalDensity(x2,y2,anchor_points,lenSents1,lenSents2,simMat,deltaX,deltaY)
        
        averageDensity=totDensity/float(len(pointsKey))

        t6=time.time()
        if verbose:
            print("\n6. Computing average density-->",t6-t5,"s.\n"),
            

        # =====> STEP 7 : filtering out low density points

        (anchor_points,filtered_x,filtered_y)=filterPoints(anchor_points,lenSents1,lenSents2,averageDensity,simMat,deltaX,deltaY)
        (anchor_points,filtered_x,filtered_y)=resolvingConflicts(anchor_points,lenSents1,lenSents2,simMat)

        if reiterateFiltering:
            (anchor_points,filtered_x,filtered_y)=filterPoints(anchor_points,lenSents1,lenSents2,averageDensity*2,simMat,int(deltaX/2),int(deltaY/2))

        t7=time.time()
        if verbose:
            print("\n7. Removing low density points-->",t7-t6,"s.\n"),

     
        #~ x=[point[0] for point in points]
        #~ y=[point[1] for point in points]
        #~ plt.axis([1,lenSents1,1,lenSents2])
        #~ plt.title(outputFilename+'.txt - filtered')
        #~ plt.scatter(x,y,c="red",s=1)                       
        #~ plt.show()
       
        # =====> STEP 8 : finding aligned intervals
        
        beginInt=(-1,-1)
        # adding last point as an anchor
        filtered_x.append(lenSents1-1)
        filtered_y.append(lenSents2-1)
        lastI=0
        lastJ=0
        intervals=[] # the array of pairs (beginInt,endInt) where beginInt and endInd are two points that define the interval
        nbInInterval=0
        
        (intervalLengthSent1,intervalLengthSent2,intervalLengthChar1,intervalLengthChar2)=(0,0,0,0)

        if detectIntervals:
            for num in range(0,len(filtered_x)):
                (i,j)=(filtered_x[num],filtered_y[num])
                localDensity=computeLocalDensity(i,j,anchor_points,lenSents1,lenSents2,simMat,deltaX,deltaY)
                densityRatio=0
                if averageDensity>0 :
                    densityRatio=localDensity/averageDensity
                # computation of the distance between (i,j) and (i,expected(j)) 
                expectedJ=lastJ+(i-lastI)*sentRatio
                vertical_deviation=abs(j-expectedJ)
                newInterval=False

                # monotony constraint : if the two previous and the two next anchors are monotonic but not the current
                # the current anchor is discarded
                if num >1 and num < len(filtered_x)-2:
                    if (filtered_x[num-2] <= filtered_x[num-1] <= filtered_x[num+1] <= filtered_x[num+2]) and \
                       (filtered_y[num-2] <= filtered_y[num-1] <= filtered_y[num+1] <= filtered_y[num+2]) and \
                       (not  (filtered_x[num-1] <= i <= filtered_x[num+1]) or \
                        not  (filtered_y[num-1] <= j <= filtered_y[num+1])) :
                            print(f"({i},{j}) is ignored (non monotonic)")
                            # the current point is skipped
                            filtered_x[num]=lastI
                            filtered_y[num]=lastJ
                            continue

                # deviated and low density point
                if (vertical_deviation > maxDistToTheDiagonal/2 or i<lastI or j<lastJ) and densityRatio < minDensityRatio:
                    # localDensity=computeLocalDensity(i,j,anchor_points,lenSents1,lenSents2,simMat,deltaX,deltaY)
                    # deviated point is removed if density is not high enough
                    print(f"({i},{j}) is ignored. Low density : {densityRatio=}")
                    # the current point is skipped
                    filtered_x[num]=lastI
                    filtered_y[num]=lastJ
                    continue
 
                # only the points that are near the diagonal are taken into account
                if vertical_deviation <= maxDistToTheDiagonal:
                    nbInInterval+=1
                else:
                    verbose and print(f"({i},{j}) is a deviating point {lastI=}, {lastJ=}, {densityRatio=}, {vertical_deviation=}")
                    
                    # considering next points to compute next deviation
                    preview_scope=2
                    if num+preview_scope<len(filtered_x):
                        (next_i,next_j)=(filtered_x[num+preview_scope],filtered_y[num+preview_scope])
                        next_expectedJ=lastJ+(next_i-lastI)*sentRatio
                        next_vertical_deviation=abs(next_j-next_expectedJ)
                        # the next point is aligned with previous point
                        if next_vertical_deviation <= maxDistToTheDiagonal:
                             verbose and print(f"({i},{j}) is ignored (next point is aligned with the previous). {vertical_deviation=}")
                              # the current point is skipped
                             filtered_x[num]=lastI
                             filtered_y[num]=lastJ
                             continue
                        else :
                            next_expectedJ=j+(next_i-i)*sentRatio
                            next_vertical_deviation=abs(next_j-next_expectedJ)
                            # if the next point is aligned with the current point, then a new interval should be created
                            if next_vertical_deviation <= maxDistToTheDiagonal and densityRatio > minDensityRatio:
                                 verbose and print(f"({i},{j}) is kept for a new interval because aligned with next points")
                                 newInterval=True
                            else:
                                verbose and print(f"({i},{j}) is ignored (next point is not aligned) {next_vertical_deviation=} {densityRatio=}")
                                # the current point is skipped
                                filtered_x[num]=lastI
                                filtered_y[num]=lastJ
                                continue
                    # if the deviating point has a high density then create a new interval
                    #~ # a new interval must be created from the deviating point
                    #~ if densityRatio > 1.5:
                        #~ verbose and print(f"({i},{j}) is kept for a new interval because of high density",densityRatio)
                        #~ newInterval=True
                    #~ else:
                        #~ verbose and print(f"({i},{j}) is ignored. {densityRatio=}")
                         #~ # the current point is skipped
                         #~ filtered_x[num]=lastI
                         #~ filtered_y[num]=lastJ
                         #~ continue
                    
                    
                #~ # computing distance
                d=math.sqrt((i-lastI)**2+(j-lastJ)**2)
                # if a there is a gap the previous interval is closed and a new interval will begin
                if d > maxGapSize and densityRatio > 1.5:
                    verbose and print(f"{d} > maxGapSize, {densityRatio=}")
                    newInterval=True
                   
                # Creating a new interval if necessary
                if newInterval:
                    endInt=(lastI,lastJ)
                    verbose and print(d,f"Closing interval ({beginInt},{endInt}) for point ({i},{j})")
                    if beginInt[0]<lastI and beginInt[1]<lastJ:
                        # to save the interval, we compute the density of selected points according to the horizontal width
                        if nbInInterval/(lastI - beginInt[0]) >= minHorizontalDensity and nbInInterval>1:
                            intervals.append((beginInt,endInt))
                            intervalLengthSent1+=lastI - beginInt[0] + 1
                            intervalLengthSent2+=lastJ - beginInt[1] + 1
                            for n in range(max(0,beginInt[0]),lastI+1):
                                intervalLengthChar1+=len(sents1[n])
                            for n in range(max(0,beginInt[1]),lastJ+1):
                                intervalLengthChar2+=len(sents2[n])
                        else:
                            if verbose:
                                print("Interval",beginInt,endInt,"has been discarded (density too low)")
                    beginInt=(i,j)
                    nbInInterval=0
                    
                    if showPlot4NewInterval:
                        min_x=max(0,i-100)
                        max_x=min(len(sents1)-1,i+100)
                        min_y=max(0,j-100)
                        max_y=min(len(sents2)-1,j+100)
                        
                        x=[point[0] for point in anchor_points if min_x <= point[0] <= max_x and  min_y <= point[1] <= max_y]
                        y=[point[1] for point in anchor_points if min_x <= point[0] <= max_x and  min_y <= point[1] <= max_y]
                        plt.axis([min_x,max_x,min_y,max_y])
                        plt.title(str(i)+","+str(j)+'=> new interval')
                        plt.scatter(x,y,c="black",s=1)                       
                        (i1,j1)=(i-deltaX/2,j-deltaX/2-deltaY/2)
                        (i1,j2)=(i-deltaX/2,j-deltaX/2+deltaY/2)
                        (i2,j3)=(i+deltaX/2,j+deltaX/2+deltaY/2)
                        (i2,j4)=(i+deltaX/2,j+deltaX/2-deltaY/2)
                        X=[i1,i1,i2,i2,i1]
                        Y=[j1,j2,j3,j4,j1]
                        plt.plot(X,Y,c="grey")
                        plt.show()
                
                lastI=i
                lastJ=j
        else:
            lastI=lenSents1-1
            lastJ=lenSents2-1
        

        t8=time.time()
        if verbose:
            print("\n8. Extracting alignable intervals-->",t8-t7,"s.\n"),       
        
    if lastI!=beginInt[0]:
        # closing last interval
        intervalLengthSent1+=lastI - beginInt[0] + 1
        intervalLengthSent2+=lastJ - beginInt[1] + 1
        for n in range(max(0,beginInt[0]),lastI+1):
            intervalLengthChar1+=len(sents1[n])
        for n in range(max(0,beginInt[1]),lastJ+1):
            intervalLengthChar2+=len(sents2[n])
        intervals.append((beginInt,(lastI,lastJ)))

    if verbose:
        print("Total interval length=",intervalLengthSent1,"+",intervalLengthSent2)
    return (filtered_x,filtered_y,intervals,intervalLengthSent1,intervalLengthSent2,intervalLengthChar1,intervalLengthChar2)


########################################################################## align function
def align(l1,l2,inputDir,file1,file2,inputFormat,outputDir,outputFormats,outputFilename="",col1=0,col2=1,printIds=False):
    global log,splitSent_regex,sentRatio,charRatio
    
    if splitSent and l1 not in splitSent_regex:
        verbose and print(f"Default regex ",splitSent_regex["default"],f"will be used for sentence segmentation in {l1}")
        splitSent_regex[l1]=splitSent_regex['default']
    if splitSent and  l2 not in splitSent_regex:
        verbose and print(f"Default regex ",splitSent_regex["default"],f"will be used for sentence segmentation in {l2}")
        splitSent_regex[l2]=splitSent_regex['default']
     
    # processing of an aligned file pair
    if verbose: 
        print("Processing",file1,"and",file2)
                
    (sents1,idSents1,lenSents1,seg2sents1,nbChars1)=readInputFile(inputDir,file1,inputFormat,col1,l1)
    (sents2,idSents2,lenSents2,seg2sents2,nbChars2)=readInputFile(inputDir,file2,inputFormat,col2,l2)
    
    # computing output file names
    if outputFilename=="":
        m=re.search(filePattern,file1)
        if m:
            name=m.group(1)
            outputFilename=name+"."+l1+"-"+l2
            outputAnchorFilename=name+".anchor."+l1+"-"+l2
        else:
            outputFilename=os.path.basename(file1)+"-"+os.path.basename(file2)
            outputAnchorFilename=file1+"-"+file2+".anchor"
    else:
        outputAnchorFilename=outputFilename+".anchor"
    
    
    ####################################################### extract candidate anchor points here !

    # =====> STEP 1-5 : extracting anchor points from similarity matrix
    
    if useNgrams:
        (points,x,y)=computePointsFromNgrams(sents1,sents2) # TODO : add simMat
    else:
        (points,x,y,simMat,embeds1,embeds2)=computePointsWithEncoder(preprocessor,encoder,sents1,sents2)

    #######################################################  extract filtered anchor points here !
    
    # =====> STEP 6-8 : filtering anchor points and extracting alignable intervals

    (filtered_x,filtered_y,intervals,intervalLengthSent1,intervalLengthSent2,intervalLengthChar1,intervalLengthChar2)=extractAnchorPoints(points,x,y,sents1,sents2,lenSents1,lenSents2,simMat)

    if adaptativeMode:
        sentRatio=intervalLengthSent2/intervalLengthSent1
        charRatio=intervalLengthChar2/intervalLengthChar1
        print(f"Adapted ratios : {sentRatio=} {charRatio=}")
        (filtered_x,filtered_y,intervals,intervalLengthSent1,intervalLengthSent2,intervalLengthChar1,intervalLengthChar2)=extractAnchorPoints(points,x,y,sents1,sents2,lenSents1,lenSents2,simMat)

    if writeIntervals and len(intervals)>0:
        outputIntervalFilename=outputAnchorFilename.replace(".anchor",".intervals")+".txt"
        f_int=open(outputIntervalFilename,mode="w",encoding="utf8")
        for interval in intervals:
            (x1,y1)=interval[0]
            (x2,y2)=interval[1]
            # here sentence num starts from 1
            f_int.write(f"{x1+1}-{x2+1}\t{y1+1}-{y2+1}\n")
        f_int.close()

    # anchor point output
    if (len(filtered_x)>0):
        if writeAnchorPoints:
            x_final=[]
            y_final=[]
            score=0
            nbScore=0
            for (x2,y2) in zip(filtered_x,filtered_y):
                if simMat[x2,y2] >= cosThresholdInOutputAnchors:
                    x_final.append(x2)
                    y_final.append(y2)
                    score+=simMat[x2,y2]
                    nbScore+=2
            if nbScore>0:
                meanScore=score/nbScore
                for outputFormat in outputFormats:
                    writeAlignedPoints(l1,l2,sents1,idSents1,sents2,idSents2,x_final,y_final,outputDir,outputAnchorFilename+"."+outputFormat,outputFormat,True,printIds,meanScore)
        
       
        # display of the points : eliminated points are red
        if printPlot:
        
            plt.axis([1,lenSents1,1,lenSents2])
            plt.autoscale()
            plt.title(outputFilename+'.txt - filtered')
            plt.scatter(x,y,c="red",s=1)
            plt.scatter(filtered_x,filtered_y,c="black",s=1)
            for interval in intervals:
                (i1,j1)=interval[0]
                (i2,j2)=interval[1]
                X=[i1,i1,i2,i2,i1]
                Y=[j1,j2,j2,j1,j1]
                plt.plot(X,Y,c="grey")
            if savePlot:
                plt.savefig(os.path.join(outputDir,outputFilename)+'.png')
            if showPlot:
                plt.show()
            plt.close()
            
        # writing intervals
        if len(intervals)>0 and writeAlignableArea:
            if not os.path.exists(outputDir):
                os.mkdir(outputDir)
     
            writeAlignable(sents1,idSents1,intervals,0,outputDir,file1+"."+outputFormat,outputFormat)
            writeAlignable(sents2,idSents2,intervals,1,outputDir,file2+"."+outputFormat,outputFormat)
            
    # If no interval is alignable
    if intervalLengthSent1==0 or intervalLengthSent2==0:
        if printLog:
            log.write(f"{outputFilename} not alignable\t{l1}={len(sents1)}\t{l2}={len(sents2)}\tmeanScore={0}\tsilence1={1:.3f}\tsilence2={1:.3f}\tcommandLine="+" ".join(sys.argv)+"\n")
        if verbose:
            print(f"{outputFilename} not alignable")
        return


    # =====> STEP 9 : extracting complete alignment using DTW
    
    if not doNotRunDTW:
        char_ratio=nbChars2/nbChars1 if charRatio==0 else charRatio
        verbose and print("Chararacter ratio=",char_ratio)
        
        (dtw_path,score)=run_dtw(encoder,sents1,sents2,intervals,filtered_x,filtered_y,simMat,embeds1,embeds2,char_ratio)
        # x_dtw and y_dtw contains a list of list of corresponding coordinates
        # eg. x_dtw=[[0],[1,2],[]]
        # eg. y_dtw=[[0],[1],[2]]
        
        x_dtw=[]
        y_dtw=[]
        nb_x=0
        nb_y=0
        
        if useShelve:
            encode_hash=embedShelve
        else:
            encode_hash={}

        # Chaining the points
        
        # adding empty pairs at the end
        (last_x,last_y)=dtw_path[-1]
        x_l=list(range(last_x+1,lenSents1-1))
        y_l=list(range(last_y+1,lenSents2-1))

        if len(x_l)>0:
            x_dtw.append(x_l)
            y_dtw.append([])
            if veryVerbose : 
                print (f"Empty pair=([{x_l}],[])")
        if len(y_l)>0:
            x_dtw.append([])
            y_dtw.append(y_l)
            if veryVerbose : 
                print (f"Empty pair=([],[{y_l}])")
        # constitution des groupes en fonctions des bornes
        for i in range(len(dtw_path)-1,-1,-1):
            if dtw_path[i] != ():
                (x,y)=dtw_path[i]
                # if the point is not empty (interval border)
                if i>=1 and dtw_path[i-1] !=():
                    (prev_x,prev_y)=dtw_path[i-1]
                    x_l=list(range(prev_x+1,x+1))
                    y_l=list(range(prev_y+1,y+1))

                    if len(x_l)>0 or len(y_l)>0:
                        x_dtw.append(x_l)
                        y_dtw.append(y_l)
                    nb_x+=len(x_l)
                    nb_y+=len(y_l)
                # if the point is the first of the interval, then use (x,y) as a simple point
                else:
                    #~ x_dtw.append([x])
                    #~ y_dtw.append([y])
                    #~ nb_x+=1
                    #~ nb_y+=1
                    # creating empty pairs for each gap
                    if i-2>=0 and dtw_path[i-2] !=() and printGap:
                        (prev_x,prev_y)=dtw_path[i-2]
                        x_l=list(range(prev_x+1,x+1))
                        y_l=list(range(prev_y+1,y+1))

                        if len(x_l)>0:
                            x_dtw.append(x_l)
                            y_dtw.append([])
                            if veryVerbose : 
                                print (f"Empty pair=([{x_l}],[])")
                        if len(y_l)>0:
                            x_dtw.append([])
                            y_dtw.append(y_l)
                            if veryVerbose : 
                                print (f"Empty pair=([],[{y_l}])")
    
        #~ print(f"first x={x},first y={y}")
        # adding first empty pair
        if printGap:
            x_l=list(range(0,x))
            y_l=list(range(0,y))
            if len(x_l)>0:
                x_dtw.append(x_l)
                y_dtw.append([])
                if veryVerbose : 
                    print (f"Empty pair=([{x_l}],[])")
            if len(y_l)>0:
                x_dtw.append([])
                y_dtw.append(y_l)
                if veryVerbose : 
                    print (f"Empty pair=([],[{y_l}])")                      

        x_dtw.reverse()
        y_dtw.reverse()
        
        # grouping points may occur here
        if lateGrouping:
            (x_dtw,y_dtw)=late_grouping(x_dtw,y_dtw,encoder,sents1,sents2,encode_hash,simMat,embeds1,embeds2,char_ratio)
    
        # writing output files
        meanScore=len(sents1)+len(sents2)-score
        if verbose and len(sents1)>0:
            meanScore=1-(score/(len(sents1)+len(sents2)))
            print(f"Average similarity={meanScore:.4f}")
        silence1=(len(sents1)-nb_x)/len(sents1)
        silence2=(len(sents2)-nb_y)/len(sents2)
        
        if printLog:
            log.write(f"{outputFilename}\t{l1}={len(sents1)}\t{l2}={len(sents2)}\tmeanScore={meanScore}\tignored1={(len(sents1)-nb_x)}\tsilence1={silence1:.3f}\tignored2={(len(sents2)-nb_y)}\tsilence2={silence2:.3f}\tcommandLine="+" ".join(sys.argv)+"\n")
            
        for outputFormat in outputFormats:
            writeAlignedPoints(l1,l2,sents1,idSents1,sents2,idSents2,x_dtw,y_dtw,outputDir,outputFilename+"."+outputFormat,outputFormat,False,printIds,meanScore)
        return meanScore


# for group [x_inf,..,x_sup], return the interval [x_inf-1,x_sup] (to use in distance_DTW)
# for group [y_inf,..,y_sup], return the interval [y_inf-1,y_sup] (to use in distance_DTW)
def calc_int(group_x,group_y):
    if len(group_x)==0:
        x_inf=0
        x_sup=0
    else:
        x_inf=group_x[0]-1
        x_sup=group_x[-1]
    if len(group_y)==0:
        y_inf=0
        y_sup=0
    else:
        y_inf=group_y[0]-1
        y_sup=group_y[-1]
    return (x_inf,x_sup,y_inf,y_sup)        
            
# apply a greedy algorithme to perform the best grouping (which increase sim between source and target)
def late_grouping(x_dtw,y_dtw,encoder,sents1,sents2,encode_hash,simMat,embeds1,embeds2,char_ratio):
    # this btree records the index of each group ordered by their gain
    gains = OOBTree()
    groups = []
    # initialisation of the groups data structure : foreach group, record x,y, and the corresponding dist
    for (group_x,group_y) in zip(x_dtw,y_dtw):
        (inf_x,sup_x,inf_y,sup_y)=calc_int(group_x,group_y)
        dist=distanceDTW(encoder,sents1,sents2,encode_hash,{},simMat,embeds1,embeds2,inf_x,sup_x,inf_y,sup_y,char_ratio,False) 
        groups.append( {'x':group_x,'y':group_y,"dist":dist} )
    
    # first iteration : for each group, the gain of similarity is computed whether grouping
    # on the left or on the right (direction indicates which direction has the best gain)
    # all the strictly positive gains are recorded in the gains btree
    for i in range(len(groups)):
        compute_gain(gains,groups,i,encoder,sents1,sents2,encode_hash,simMat,embeds1,embeds2,char_ratio)

    if len(gains)>0:
        best_gain=gains.maxKey()
    else:
        best_gain=0
    # while best grouping produce a positive gain in similarity 
    while best_gain > 0:
        i=gains[best_gain][-1]
        
        group_x=groups[i]['x']
        group_y=groups[i]['y']
        
        # group i with next group
        if groups[i]['direction']==1:
            next_i=next(groups,i)
            verbose and print(f"group {i} with next {next_i} :",groups[i]['newX'],groups[i]['newY'])
            if next_i!=-1:
                # the next group is first "deleted" : dist is set to -1, and x and y are set to []
                groups[next_i]['dist']=-1
                groups[next_i]['x']=[]
                groups[next_i]['y']=[]
            else:
                print("Wrong direction")
                break
        elif groups[i]['direction']==-1:
        # group i with previous group
            prev_i=prev(groups,i)
            verbose and print(f"group {i} with prev {prev_i} :",groups[i]['newX'],groups[i]['newY'])
            if prev_i!=-1:
                # the prev group is first "deleted" : dist is set to -1, and x and y are set to []
                groups[prev_i]['dist']=-1
                groups[prev_i]['x']=[]
                groups[prev_i]['y']=[]
            else:
                print("Wrong direction")
                break
        else:
            print("No direction",i,groups[i])
            break

        # setting the new group with the recorded merging
        groups[i]['x']=groups[i]['newX']
        groups[i]['y']=groups[i]['newY']
        groups[i]['dist']=groups[i]['newDist']
        
        # update of the gain, on the left and on the right (after the prev group or the next_group which are "deleted")
        compute_gain(gains,groups,i,encoder,sents1,sents2,encode_hash,simMat,embeds1,embeds2,char_ratio)
        
        # update gain on the left and right side
        prev_i=prev(groups,i)
        if prev_i!=-1:
            compute_gain(gains,groups,prev_i,encoder,sents1,sents2,encode_hash,simMat,embeds1,embeds2,char_ratio)
            
        next_i=next(groups,i)
        if next_i!=-1:
            compute_gain(gains,groups,next_i,encoder,sents1,sents2,encode_hash,simMat,embeds1,embeds2,char_ratio)

        # computing best gain for next iteration
        if len(gains)>0:
            best_gain=gains.maxKey()
        else:
            best_gain=0

    # returning final groups
    x_dtw=[]
    y_dtw=[]
    for i,group in enumerate(groups):
        if groups[i]['dist']!=-1:
            x_dtw.append(groups[i]['x'])
            y_dtw.append(groups[i]['y'])
    return (x_dtw,y_dtw)
    

# compute the gain when grouping on the left (direction=-1) or on the right (direction=1) side
# and record the corresponding merged groups and distance
def compute_gain(gains,groups,i,encoder,sents1,sents2,encode_hash,simMat,embeds1,embeds2,char_ratio):
    group_x=groups[i]['x']
    group_y=groups[i]['y']
    dist=groups[i]['dist']

    # removing i for old_gain if any
    if 'gain' in groups[i]:
        old_gain=groups[i]['gain']
        if old_gain>0 and i in gains[old_gain]:
            gains[old_gain].remove(i)
            # removing the old gain key if necessary
            if len(gains[old_gain])==0:
                del(gains[old_gain])

    # no gain is computed for empty groups
    if len(group_x)==0 or len(group_y)==0:
        groups[i]['gain']=0
        return
    
    prev_i=prev(groups,i)
    prev_gain=0
    if prev_i!=-1:
        prev_group_x=groups[prev_i]['x']
        prev_group_y=groups[prev_i]['y']
        no_empty= len(prev_group_x)>0 and len(prev_group_y)>0
        new_group_x1=prev_group_x+group_x
        new_group_y1=prev_group_y+group_y
        (inf_x,sup_x,inf_y,sup_y)=calc_int(new_group_x1,new_group_y1)
        prev_dist=distanceDTW(encoder,sents1,sents2,encode_hash,{},simMat,embeds1,embeds2,inf_x,sup_x,inf_y,sup_y,char_ratio,False) 
        prev_gain=dist-prev_dist
        if no_empty:
            prev_gain -= penalty_n_n
        else:
            prev_gain += penalty_0_n
        #~ print(i,"prev",no_empty,prev_gain,new_group_x1,new_group_y1,dist,prev_dist,prev_gain)

    next_i=next(groups,i)
    next_gain=0
    if next_i!=-1:
        next_group_x=groups[next_i]['x']
        next_group_y=groups[next_i]['y']
        no_empty= len(next_group_x)>0 and len(next_group_y)>0
        new_group_x2=group_x+next_group_x
        new_group_y2=group_y+next_group_y
        (inf_x,sup_x,inf_y,sup_y)=calc_int(new_group_x2,new_group_y2)
        next_dist=distanceDTW(encoder,sents1,sents2,encode_hash,{},simMat,embeds1,embeds2,inf_x,sup_x,inf_y,sup_y,char_ratio,False) 
        next_gain=dist-next_dist
        if no_empty:
            next_gain -= penalty_n_n
        else:
            next_gain += penalty_0_n
        #~ print(i,"next",no_empty,next_gain,new_group_x2,new_group_y2,dist,next_dist,next_gain)

    if next_gain > prev_gain and next_gain > 0:
        groups[i]['gain']=next_gain
        groups[i]['direction']=1
        groups[i]['newX']=new_group_x2
        groups[i]['newY']=new_group_y2
        groups[i]['newDist']=next_dist
        gain=next_gain
    elif prev_gain > 0 :
        groups[i]['gain']=prev_gain
        groups[i]['direction']=-1
        groups[i]['newX']=new_group_x1
        groups[i]['newY']=new_group_y1
        groups[i]['newDist']=prev_dist
        gain=prev_gain
    else:
        groups[i]['gain']=0
        groups[i]['direction']=0
        gain=0
        
    # updating gains btree
    if gain>0:
        if not gains.has_key(gain):
            gains[gain]=[]
        if i not in gains[gain]:
            gains[gain].append(i)

    return gain

# search for the previous non deleted group before group i 
def prev(groups,i):
    if i==0:
        return -1
    i-=1
    while i>0 and groups[i]["dist"]==-1 :
        i-=1
    if groups[i]["dist"]==-1:
        return -1
    else:
        return i
        
# search for the next non deleted group after group i
def next(groups,i):
    if i==len(groups)-1:
        return -1
    i+=1
    while i<len(groups)-1 and groups[i]["dist"]==-1 :
        i+=1
    if groups[i]["dist"]==-1:
        return -1
    else:
        return i


# Run the Dynamic time warping algorithm (Viterbi) by computing all the paths
# from each anchor points (the paths must not deviate from these anchors points
# at a distance lower than dtwBeam) 

def run_dtw(encoder,sents1,sents2,intervals,filtered_x,filtered_y,simMat,embeds1,embeds2,char_ratio):
    global useShelve
    global embedShelve
    
    path_hash={}
    dist_hash={"-2--1;-2--1":0} # for the point (-1,-1), the lower bound
    
    if useShelve:
        encode_hash=embedShelve
    else:
        encode_hash={}
    
    # initialization for the NULL path
    x_first=intervals[0][0][0]
    y_first=intervals[0][0][1]
    path_hash[f"{x_first}-{y_first}"]=[[[-1,-1]],0]
    
    print( f"Init : dtw from ",intervals[0][0]," to ",intervals[-1][1])
    
    lastBestPath=[[intervals[0][0][0]-1,intervals[0][0][1]-1]]
    lastBestScore=0
    
    t8=time.time()
    # process each alignable intervals
    for interval in intervals:
        (x_begin,y_begin)=interval[0]
        (x_end,y_end)=interval[1]
        key_xy= f"{x_begin}-{y_begin}"
        coeff_y_per_x=(y_end-y_begin)/(x_end-x_begin)
        
        # these dict allow to drive the paths near the anchor points that are located IN the interval
        x_2_y={}
        y_2_x={}
        for i in range(len(filtered_x)):
            x=filtered_x[i]
            y=filtered_y[i]
            if x<x_begin:
                continue
            if x>x_end:
                break
            if y<y_begin or y>y_end:
                continue
            x_2_y[x]=y
            y_2_x[y]=x

        # Bridging the gap between alignable intervals
        # if there is a gap between the last point in path and the first point in current interval, add an empty point () in the path
        if key_xy not in path_hash:
            (lastI,lastJ)=lastBestPath[-1]
            if verbose:
                print(f"Inserting gap between ({lastI},{lastJ}) and ({x_begin},{y_begin})")
            lastBestPath.append(()) # an empty point indicate a break in the path
            lastBestPath.append((x_begin-1,y_begin-1))
            path_hash[key_xy]=[lastBestPath,lastBestScore]

        
        # now run the DTW search between each anchor point in the interval
        # the path are computed recursively, but in order to minimize the recursive depth, the
        # dtw hash is progressively filled by calling the function point by point
        previous1_x=x_begin
        previous1_y=y_begin
        for x in range(x_begin,x_end+1):
            localBeam=dtwBeam
            
            # case 1 : if (x,y) is an anchor point
            if x in x_2_y:
                y=x_2_y[x]
                if verbose:
                    print(f"Anchor point {x},{y}")
                    
                # if (x,y) is too far from the interval diagonal, it is discarded
                deviation=0
                if y>=y_begin:
                    deviation=abs((y-y_begin)/(y_end-y_begin) - (x-x_begin)/(x_end-x_begin))
                else:
                    continue
               
                # First condition : 1/ deviation > localDiagBeam
                if (deviation > localDiagBeam and deviation*(y_end-y_begin)>dtwBeam) :
                    del x_2_y[x]
                    if y in y_2_x:
                        del y_2_x[y]
                    if verbose:
                        print( f"deviation*(y_end-y_begin)= {deviation*(y_end-y_begin)} - Anchor point ({x},{y}) is too far from the interval diagonal - point has been discarded!")
                    continue
                # Second condition : 2/ the ratio between deltaX and deltaY exceeds 4 (1-4 or 4-1 grouping is the max allowed)
                if (noEmptyPair and (min(y-previous1_y,x-previous1_x)==0 or max(y-previous1_y,x-previous1_x)/min(y-previous1_y,x-previous1_x)>4)) :
                    del x_2_y[x]
                    if y in y_2_x:
                        del y_2_x[y]
                    if verbose:
                        print( f"Deviating anchor point ({x},{y}) is too close from the preceding - point has been discarded!")
                    continue
                
                # Processing of the gaps (taking into account non monotony) :
                # the localBeam is recomputed according to the deviation of the current anchor point
                # from the previous anchor point - according to x axis (previous1_x,previous1_y) 
                # and y axis (previous2_x,previous2_y) - the max deviation is taken into account

                if (previous1_y < y) and abs((y-previous1_y) - int((x-previous1_x)*coeff_y_per_x)) > dtwBeam :
                    localBeam=abs((y-previous1_y) - int((x-previous1_x)*coeff_y_per_x))+dtwBeam+1
                    print( f"Applying local margin {localBeam} for point : ({x},{y}) previous1=({previous1_x},{previous1_y}) with coeff={coeff_y_per_x}")
                    
                previous2_y=y-1
                # looking for previous point according to y
                while previous2_y > y_begin and previous2_y not in y_2_x:
                    previous2_y-=1
                if previous2_y in y_2_x:
                    previous2_x=y_2_x[previous2_y]
                    if (previous2_x < x) and abs((y-previous2_y) - int((x-previous2_x)*coeff_y_per_x)) > localBeam :
                        localBeam=abs((y-previous2_y) - int((x-previous2_x)*coeff_y_per_x))+dtwBeam+1
                        print( f"Applying local margin {localBeam} for point : ({x},{y}) previous2=({previous2_x},{previous2_y})  with coeff={coeff_y_per_x}")
                if veryVerbose:
                    print( f"Running DTW for the point : ({x},{y}) - elapsed from (1,1) =",time.time()-t8,"s.")
                
                (path,dist)=dtw(encoder,sents1,sents2,encode_hash,path_hash,dist_hash,x_2_y,y_2_x,simMat,embeds1,embeds2,x,y,x_begin,y_begin,localBeam,char_ratio)
                if dist==infinite and verbose: 
                    print( f"Infinite distance from : ({x},{y})")
                    # initiating a new interval starting from x,y
                    x_begin=x
                    y_begin=y
                    key_xy= f"{x_begin}-{y_begin}"
                    # here creation of a copy of lastBestPath, and addition of the breakpoint
                    lastBestPath=lastBestPath[:]
                    lastBestPath.append(()) # an empty point indicate a break in the path
                    lastBestPath.append((x_begin-1,y_begin-1))
                    path_hash[key_xy]=[lastBestPath,lastBestScore]
                    #~ sys.exit()
                else:
                    lastBestPath=path
                    lastBestScore=dist
                    
                if veryVerbose:
                    print(f"Distance->{dist}")
                previous1_x=x
                previous1_y=y

        (lastBestPath,lastBestScore)=path_hash[f"{previous1_x}-{previous1_y}"]
    
    # chaining with the end of the text
    last_x=len(sents1)-1
    last_y=len(sents2)-1
    if (last_x-x)+(last_y-y)<200:
        if verbose:
            print( f"Last point ({last_x},{last_y})")
        dtw(encoder,sents1,sents2,encode_hash,path_hash,dist_hash,x_2_y,y_2_x,simMat,embeds1,embeds2,last_x,last_y,x_end,y_end,dtwBeam,char_ratio)
    # if last point has not been discarded
    score=infinite
    if f"{last_x}-{last_y}" in path_hash:
        (best_path,score)=path_hash[f"{last_x}-{last_y}"]
    # the last interval is used instead
    if score==infinite:
        (best_path,score)=path_hash[f"{previous1_x}-{previous1_y}"]
    
    t9=time.time()
    if verbose:
        print( f"\n9. Elapsed time for complete DTW-->",t9-t8,"s.\n")

    return (best_path,score)


# Compute the bestpath (a list of [I,J] pairs) and the corresponding score (the minimum distance)
# The current point correspond to the interval between (infI,inJ) excluded
def dtw(encoder,sents1,sents2,encode_hash,path_hash,dist_hash,x_2_y,y_2_x,simMat,embeds1,embeds2,i,j,x_begin,y_begin,localBeam,char_ratio):
    global verbose
    global dtwBeam
    # at each recursion step, localBeam decreases to the dtwBeam floor value
    localBeam=max(localBeam-localBeamDecay,dtwBeam)
    
    # The hash path_hash records the result for already computed path, in order to reduce recursivity
    dtw_key=str(i)+"-"+str(j)

    if dtw_key in path_hash:
        return path_hash[dtw_key]

    # if point is two far from the corresponding anchor point on vertical or horizontal axis, the path is discarded (dist=infinite)
    if i in x_2_y and abs(x_2_y[i]-j) > localBeam:
        #~ print (f"Rejection of point ({i},{j}) too far from ({i},{x_2_y[i]}) with localBeam={localBeam}")
        return ([],infinite)
    if j in y_2_x and abs(y_2_x[j]-i) > localBeam:
        #~ print (f"Rejection of point ({i},{j}) too far from ({y_2_x[j]},{j}) with localBeam={localBeam}")     
        return ([],infinite)
    
    # end of recursivity if the current coordinates reach the @inf point : the path must end here
    if i<x_begin-1 or j<y_begin-1:
        return ([],infinite)

    path_by_group={}
    dist_by_group={}
    for group in allowed_groups:
        (path_by_group[group],dist_by_group[group])=dtw(encoder,sents1,sents2,encode_hash,path_hash,dist_hash,x_2_y,y_2_x,simMat,embeds1,embeds2,i-group[0],j-group[1],x_begin,y_begin,localBeam,char_ratio)
        dist_by_group[group]+= distanceDTW(encoder,sents1,sents2,encode_hash,dist_hash,simMat,embeds1,embeds2,i-group[0],i,j-group[1],j,char_ratio) # interval ]i-group[0];i] ]j-group[1];j] 

    best_group=None
    min_dist=infinite
    for group in allowed_groups:
        if dist_by_group[group]<min_dist:
            min_dist=dist_by_group[group]
            best_group=group
    if best_group!=None:
        path=path_by_group[best_group][:] # warning here, create a copy !
        path.append([i,j])
        path_hash[dtw_key]=[path,min_dist]
        return (path,min_dist)
    
    path_hash[dtw_key]=[[],infinite]
    return ([],infinite)
    
# computing the distance as 1-cosinus
# for empty aligning, dist is equal to distNull which should be near to 1
# when the similarity is below a given threshold (sim_threshold), the dist is fixed to 1 (in order to force using 1-0 or 0-1 pairing)

def distanceDTW(encoder,sents1,sents2,encode_hash,dist_hash,simMat,embeds1,embeds2,inf_i,i,inf_j,j,char_ratio,use_coeff=True):
    # if the distance has already been stored in dist_hash
    key=str(inf_i)+"-"+str(i)+";"+str(inf_j)+"-"+str(j)
    if key in dist_hash:
        return dist_hash[key]
        
    # coeff indicates the total number of segments (for both language) involved in the alignment
    coeff=1
    penalty=penalty_n_n

    # case of relations 1-0 et 0-1
    if inf_i==i or inf_j==j:
        return distNull * coeff
    
    if i < 0 or j < 0 or inf_i < -2 or inf_j < -2:
        return infinite
    
    coeff=2
    if useEncoder:
        # similarity are computed for sentence group
        # case of relations 1-1
        if inf_i==i-1 and inf_j==j-1:
            sim=simMat[i,j]
            if use_coeff:
                penalty=0
        # case of relations n-n
        else :
            # calculate embed_i
            if inf_i==i-1 :
                embed_i=embeds1[i][:]
                len_i=len(sents1[inf_i+1])
            else :
                sent_i=sents1[inf_i+1]
                for coord_i in range(inf_i+2,i+1):
                    sent_i+=" "+sents1[coord_i]
                    if use_coeff:
                        coeff+=1
                len_i=len(sent_i)
                if sent_i not in encode_hash:
                    embed_i=encoder.encode([sent_i])
                    embed_i = embed_i / np.linalg.norm(embed_i) # normalize
                    encode_hash[sent_i]=embed_i
                else:
                    embed_i=encode_hash[sent_i]
            # calculate embed_j
            if inf_j==j-1 :
                embed_j=embeds2[j][:]
                len_j=len(sents2[inf_j+1])
            else :  
                sent_j=sents2[inf_j+1]
                for coord_j in range(inf_j+2,j+1):
                    sent_j+=" "+sents2[coord_j]
                    if use_coeff:
                        coeff+=1
                len_j=len(sent_j)
                if sent_j not in encode_hash:
                    embed_j=encoder.encode([sent_j])
                    embed_j= embed_j / np.linalg.norm(embed_j) # normalize
                    encode_hash[sent_j]=embed_j
                else:
                    embed_j=encode_hash[sent_j]
            sim=float(np.matmul(embed_i, np.transpose(embed_j)))
    else:
        # similarity are computed with vector addition
        # case of relations 1-1 : no penalty
        if inf_i==i-1 and inf_j==j-1:
            penalty=0
        
        embed_i=embeds1[inf_i+1][:]
        embed_j=embeds2[inf_j+1][:]
        
        len_i=len(sents1[inf_i+1])
        for coord_i in range(inf_i+2,i+1):
            len_i+=len(sents1[coord_i])
            embed_i=np.add(embed_i,embeds1[coord_i])
            if use_coeff:
                coeff+=1

        len_j=len(sents2[inf_j+1])
        for coord_j in range(inf_j+2,j+1):
            len_j+=len(sents2[coord_j])
            embed_j=np.add(embed_j,embeds2[coord_j])
            if use_coeff:
                coeff+=1
        
        embed_i = embed_i / np.linalg.norm(embed_i) # normalize
        embed_j = embed_j / np.linalg.norm(embed_j) # normalize
        sim=np.matmul(embed_i, np.transpose(embed_j))

    # compute the similarity with neighbouring sentences and substract it to the global sim
    if not noMarginPenalty:
        nb=0
        if inf_j>=0:
            left_embed_j=embeds2[inf_j][:]
            left_sim_j=np.matmul(embed_i, np.transpose(left_embed_j))
            nb+=1
        else:
            left_sim_j=0
        if j+1<len(embeds2):
            right_embed_j=embeds2[j+1][:]
            right_sim_j=np.matmul(embed_i, np.transpose(right_embed_j))
            nb+=1
        else:
            right_sim_j=0
        neighbour_sim_j=(left_sim_j+right_sim_j)/nb
        
        nb=0
        if inf_i>=0:
            left_embed_i=embeds1[inf_i][:]
            left_sim_i=np.matmul(left_embed_i, np.transpose(embed_j))
            nb+=1
        else:
            left_sim_i=0
        if i+1<len(embeds1):
            right_embed_i=embeds1[i+1][:]
            right_sim_i=np.matmul(right_embed_i, np.transpose(embed_j))
            nb+=1
        else:
            right_sim_i=0
        neighbour_sim_i=(left_sim_i+right_sim_i)/nb
        
        average_neighbour_sim=(neighbour_sim_i+neighbour_sim_j)/2
        sim-=coeff_neighbour_sim*average_neighbour_sim

    # for empty sentences
    if len_i*len_j==0:
        return distNull * coeff

    dist=1-sim
    if use_coeff:
        dist += penalty * coeff

    dist = (1-coeff_sent_len)*dist + coeff_sent_len*lenPenalty(len_i*char_ratio,len_j)

    dist *= coeff
    dist_hash[key]=dist
    return dist

# cf Bertalign
def lenPenalty(len1,len2):
    min_len = min(len1,len2)
    max_len = max(len1,len2)
    return 1-np.log2(1 + min_len / max_len)



#************************************************************************* MAIN
if __name__ == "__main__":
    t0=time.monotonic()

    # processing a unic pair of files
    if inputFile1 and inputFile2:
        align(l1,l2,inputDir,inputFile1,inputFile2,inputFormat,outputDir,outputFormats,outputFilename,col1=col1,col2=col2,printIds=printIds)
    # processing a full directory
    else :
        if verbose:
            print("Processing directory",inputDir)
        # reading a tsv file with pairs fileName1 tab fileName2
        if inputFileList:
            f=open(inputFileList,encoding="utf8")
            files1=[]
            files2=[]
            for line in f:
                # skipping comments
                if line[0]!="#":
                    values=line.strip().split("\t")

                    if len(values)==2:
                        f1=values[0]
                        f2=values[1]
                        files1.append(f1)
                        files2.append(f2)
            f.close()
            if verbose:
                print("Files to process",list(zip(files1,files2)))
            for file1,file2 in zip(files1,files2):
                outputFilename=""
                if filePattern.match(file1):
                    outputFilename= filePattern.match(file1).group(1)
                    l1= filePattern.match(file1).group(2)
                    l2= filePattern.match(file2).group(2)
                align(l1,l2,inputDir,file1,file2,inputFormat,outputDir,outputFormats,outputFilename,col1=col1,col2=col2,printIds=printIds)
        else:
            files=[f for f in os.listdir(inputDir) if filePattern.match(f) ] # and re.search(inputFormat+"$",f,re.I)]
            files1=[f for f in files if filePattern.match(f).group(2)==l1]
            files2=[f for f in files if filePattern.match(f).group(2)!=l1 and (filePattern.match(f).group(2)==l2 or l2=="*")]
            if verbose:
                print("Files to process",files1)
            # processing input files
            for file1 in files1:
                m=filePattern.match(file1)
                name=m.group(1)
                for file2 in files2:
                    m=filePattern.match(file2)
                    if m.group(1)==name:
                        l2=m.group(2)
                        align(l1,l2,inputDir,file1,file2,inputFormat,outputDir,outputFormats,outputFilename="",col1=col1,col2=col2,printIds=printIds)
    if  verbose:
        print ("Terminated in",time.monotonic()-t0,"s.")
        
    if useShelve:
        embedShelve.close() 
    if printLog:
        log.close()
