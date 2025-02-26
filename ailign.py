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
The first filtering corresponds to the filter_points() function. The first filter is based on a calculation of the density of candidates around each candidate point. This density is not calculated in a square centered around the point, but rather in a corridor centered on the diagonal passing through the point (the alignment path generally follows this diagonal). The width of this corridor corresponds to the deltaY parameter. The length of this corridor corresponds to the deltaY parameter. The number of candidate points divided by the size of this space gives a density value. If this density, divided by the average density of all candidate points, is greater than a certain ratio (minDensityRatio, typically around 0.5) then the point is retained.
The second filter, which corresponds to the resolving_conflicts() function, focuses on resolving conflicts on the vertical and horizontal axes respectively - when for the same x-coordinate there are several points with different y-coordinates, and conversely, when for the same y-coordinate there are several points with different x-coordinates - these cases only arise if KBest is greater than 1. Competitors are eliminated on the basis of density: only the point with the best density along its diagonal is retained.
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
from lxml import etree
import shelve
import math

from datetime import datetime
from BTrees.OOBTree import OOBTree

import numpy as np

import matplotlib.pyplot as plt
from lexical_alignment import chunk_alignment, word_alignment
#import torch

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
parser.add_argument('--l2', type=str, help='The target language (ISO : ex. "fr" for French, "*" for any)', default='fr')
parser.add_argument('-i', '--inputFormat', help='Format of the input (txt, arc, ces, json, tsv, xml-conll, xml)',
                    default="txt")
parser.add_argument('--xmlGuide', nargs='+', type=str, help='List of markups that should be read in the XML input',
                    default=["s"])
parser.add_argument('--anchorTag',type=str, help='Tag that defines prealigned anchors in XML input (eg. "anchor" or "p")',
                    default="")
           
parser.add_argument('--col1', help='For TSV format, indicate the column of l1', type=int, default=0)
parser.add_argument('--col2', help='For TSV format, indicate the column of l2', type=int, default=1)
parser.add_argument('-o', '--outputFormats', nargs='+', type=str,
                    help='Formats of the output (TXT, TXT2, CES, ARC, XML, TSV, TSV2, BERTALIGN)',
                    default=["txt", "tmx", "ces"])
parser.add_argument('--collectionName', help='for TSV2 format (Lexicoscope) name of the collection', default="")
parser.add_argument('--addAnchor', help='Add anchor in xml files', action="store_true", default=False)
parser.add_argument('--direction', type=str, help='The aligning direction for anchors: "1<->2","1->2","2->1"',
                    default='1<->2')
parser.add_argument('--inputFile1', type=str, help='The l1 input file to process', default='corpus/troiscochons_en.txt')
parser.add_argument('--inputFile2', type=str, help='The l2 input file to process', default='corpus/troiscochons_fr.txt')
parser.add_argument('--fileId1', type=str, help='The id prefix of file1 in xml anchors', default='')
parser.add_argument('--fileId2', type=str, help='The id prefix of file2 in xml anchors', default='')
parser.add_argument('--inputFileList', type=str, help='A tsv file with corresponding filenames separated by tab',
                    default='')
parser.add_argument('--inputDir', type=str, help='The directory to process', default='.')
parser.add_argument('--outputDir', type=str, help='The directory to save output files', default='')
parser.add_argument('--outputFilename', type=str, help='The output filename (optional), without format extension',
                    default='./output.l1-l2')
parser.add_argument('-f', '--filePattern', type=str,
                    help='The pattern of the files that should be processed. A capturing group such as (.*) should capture the common prefix between aligned files.',
                    default=r'([^\\/]*)[.](\w\w\w?)[.]\w+$')
parser.add_argument('--writeAnchorPoints', help='Write anchor points', action="store_true", default=False)
parser.add_argument('--writeSegmentedInput', help='Write sentence segmented input files in txt format',
                    action="store_true", default=False)
parser.add_argument('--writeIntervals', help='Write aligned intervals (as corresponding sentence numbers)',
                    action="store_true", default=False)
parser.add_argument('--printIds', help='Print IDs in txt output', action="store_true", default=False)
parser.add_argument('--splitSent', help='Split the txt segments into sentences', action="store_true", default=False)
parser.add_argument('--splitSentRegex', type=str, help='Regex to split sentences', default="")

parser.add_argument('--useSentenceSegmenter',
                    help='Use the trankit sentence segmenter for txt input (instead of regex segmenter)',
                    action="store_true", default=False)
parser.add_argument('--mergeLines', help='Merge lines until a line ends with a separator for txt input',
                    action="store_true", default=False)
parser.add_argument('--adaptativeMode',
                    help='Using interval detection, compute estimated sentRatio and charRatio, and reiterate filtering.',
                    action="store_true", default=False)

# special arguments for output control
parser.add_argument('-v', '--verbose', help='Verbose messages', action="store_true")
parser.add_argument('-w', '--writeAlignableArea', help='Write alignable area files', action="store_true",
                    default=False)
parser.add_argument('-V', '--veryVerbose', help='Very verbose messages', action="store_true")
parser.add_argument('--savePlot', help='Save scatter plot in a png file', action="store_true", default=False)
parser.add_argument('--showPlot', help='Show scatter plot (with a pause during execution)', action="store_true",
                    default=False)
parser.add_argument('--showSimMat', help='Show heat map for similarity matrix', action="store_true",
                    default=False)

# controlling stage 1 and 2
parser.add_argument('--detectIntervals', help='Detect alignable interval using anchor points.', action="store_true",
                    default=False)
parser.add_argument('-u', '--useNgrams', help='Use ngrams to extract points', action="store_true", default=False)
parser.add_argument('-r', '--doNotRunDTW', help='Perform only first step without DTW algorithm)', action="store_true",
                    default=False)
parser.add_argument('--lateGrouping',
                    help='Run DTW algorithm with only 1-1 pairing, then, group the contiguous points with lateGrouping method (greedy algorithm)',
                    action="store_true", default=False)
parser.add_argument('--noMarginPenalty',
                    help='Do not compute the similarity with neighbouring sentences, and substract the neighbouring similarity to the bead similarity)',
                    action="store_true", default=False)

# controlling anchor points building and filtering
# (important parameters are : cosThreshold, kBest, deltaX, minDensityRatio)
parser.add_argument('--embedModel', type=str, help='Choose embedding model : sbert or laser or labse-keras or stsb-xlm-r-multilingual',
                    default="sbert")
parser.add_argument('--modelName', type=str, help='Choose sbert model name (default=sentence-transformers/LaBSE)',
                    default="sentence-transformers/LaBSE")
parser.add_argument('-l', '--cosThreshold', type=float,
                    help='The minimum similarity for labse vectors to yield one point', default=0.4)
parser.add_argument('--cosThresholdInOutputAnchors', type=float, help='The minimum similarity for final anchor points',
                    default=0.5)
parser.add_argument('--ngram', type=int, help='The ngram size', default=4)
parser.add_argument('-d', '--diceThreshold', type=float, help='The minimum dice score to yield a candidate point',
                    default=0.05)
parser.add_argument('--margin', type=float,
                    help='Margin used to eliminate sentences that have too close neighbours on the vertical or horizontal axis',
                    default=0.05)
parser.add_argument('-k', '--kBest', type=int,
                    help='Number of the best coordinates for each line ore column to keep when creating points',
                    default=4)
parser.add_argument('-x', '--deltaX', type=int, help='Local space definition : +/-delta X on horizontal axis',
                    default=20)
parser.add_argument('-y', '--deltaY', type=int, help='Local space definition : +/-delta Y on vertical axis', default=3)
parser.add_argument('-H', '--minHorizontalDensity', type=float,
                    help='The minimal horizontal density in a interval to be kept in the final result', default=0.05)
parser.add_argument('-m', '--maxDistToTheDiagonal', type=int,
                    help='The maximal distance to the diagonal (inside a given interval) for a point to be taken into account in the horizontal density',
                    default=20)
parser.add_argument('-D', '--minDensityRatio', type=float,
                    help='The minimal local density ratio (reported to the average local density) to keep a candidate point',
                    default=0.3)
parser.add_argument('-g', '--maxGapSize', type=int,
                    help='The maximal distance between to consecutive points in the same interval', default=100)
parser.add_argument('--diagBeam', type=float,
                    help='A real number in the range 0-1 which indicate the max distance of anchor points to the diagonal (vertically), in proportion (1 indicates that the whole search space is used',
                    default=1)
parser.add_argument('--localDiagBeam', type=float,
                    help='A real number in the range 0-1 which indicate the max distance of anchor points to the diagonal of each alignable interval (vertically), in proportion (1 indicates that the whole search space is used',
                    default=0.2)
parser.add_argument('--sentRatio', type=float,
                    help='The sentence ratio is used during anchor point filtering. Normally computed automatically, may be forced when texts have very different length.',
                    default=0)
parser.add_argument('--charRatio', type=float,
                    help='The character ratio is used during final aligning when groups of sentences are paired. Normally computed automatically, may be forced when texts have very different length.',
                    default=0)
parser.add_argument('--reiterateFiltering', help='Filter the anchor points according to density twice',
                    action="store_true", default=False)

# controlling DTW algorithm
parser.add_argument('--dtwBeam', help='Max dist to the anchor point in DTW algorithm', type=int, default=3)
parser.add_argument('--localBeamDecay', help='Decreasing value of localBeam at each recursion step', type=float,
                    default=0.5)
parser.add_argument('--distNull', help='Default distance for null correspondance', type=float, default=1)
parser.add_argument('--noEmptyPair', help='No 1-0 or 0-1 pairing', action="store_true", default=False)
parser.add_argument('--no2_2Group', help='No 2-2 pairing', action="store_true", default=False)
parser.add_argument('--penalty_n_n', help='Penalty score given for each n-n grouping', type=float, default=0.06)
parser.add_argument('--penalty_0_n',
                    help='Penalty score given for each 0-n (or n-0) grouping (only used in lateGrouping)', type=float,
                    default=0.15)

parser.add_argument('--wordAlignment', help='Run the word alignment script', action="store_true", default=False)
parser.add_argument('--chunkAlignment', help='Run the chunk alignment script', action="store_true", default=False)

# other : persistance of embeddings
parser.add_argument('--useShelve', help='Save the embeddings in shelve (in order to quick up the next run)',
                    action="store_true", default=False)


args = parser.parse_args()

# generic parameters
# arguments of the ailign function
# align(l1,l2,input_dir,file1,file2,inputFormat,outputDir,outputFormats,output_filename="",col1=0,col2=1,printIds=False,file_id1="",file_id2="",add_anchor=False):
l1 = args.l1
l2 = args.l2
input_dir = args.inputDir
input_file1 = args.inputFile1
input_file2 = args.inputFile2
input_format = args.inputFormat  # 'txt','arc','json'
output_formats = args.outputFormats
collection_name = args.collectionName
output_file_name = args.outputFilename

output_dir = args.outputDir
# if no output dir, we take the path of outputFilename
if output_dir=="":
    output_dir=os.path.split(output_file_name)[0]
    
col1 = args.col1
col2 = args.col2
print_ids = args.printIds
file_id1 = args.fileId1
file_id2 = args.fileId2
add_anchor = args.addAnchor

params = {}

params['inputFileList'] = args.inputFileList
params['verbose'] = args.verbose
params['detectIntervals'] = args.detectIntervals
params['writeAlignableArea'] = args.writeAlignableArea
params['writeAnchorPoints'] = args.writeAnchorPoints
params['writeSegmentedInput'] = args.writeSegmentedInput
params['writeIntervals'] = args.writeIntervals
params['direction'] = args.direction
params['veryVerbose'] = args.veryVerbose
params['filePattern'] = re.compile(args.filePattern)
params['savePlot'] = args.savePlot
params['showPlot'] = args.showPlot
params['showSimMat'] = args.showSimMat
params['xmlGuide'] = args.xmlGuide
params['anchorTag'] = args.anchorTag
params['splitSent'] = args.splitSent
params['splitSentRegex'] = args.splitSentRegex
params['useSentenceSegmenter'] = args.useSentenceSegmenter
params['mergeLines'] = args.mergeLines
params['adaptativeMode'] = args.adaptativeMode
params['useNgrams'] = args.useNgrams
params['doNotRunDTW'] = args.doNotRunDTW
params['noMarginPenalty'] = args.noMarginPenalty
params['lateGrouping'] = args.lateGrouping
params['noEmptyPair'] = args.noEmptyPair
params['no2_2Group'] = args.no2_2Group
params['penalty_n_n'] = args.penalty_n_n
params['penalty_0_n'] = args.penalty_0_n
params['charRatio'] = args.charRatio
params['sentRatio'] = args.sentRatio

# sentence encoder method parameters
params['embedModel'] = args.embedModel
params['modelName'] = args.modelName
params['cosThreshold'] = args.cosThreshold
params['cosThresholdInOutputAnchors'] = args.cosThresholdInOutputAnchors
params['dtwBeam'] = args.dtwBeam
params['localBeamDecay'] = args.localBeamDecay
params['distNull'] = args.distNull

# ngram identification
params['ngram'] = args.ngram  # ngram size
params['diceThreshold'] = args.diceThreshold  # min dice to add a candidate point

# anchor point filtering parameters
params['deltaX'] = args.deltaX  # local space definition : +/-delta X on horizontal axis
params['deltaY'] = args.deltaY  # local space definition : +/-delta Y on vertical axis
params[
    'minDensityRatio'] = args.minDensityRatio  # the minimal local density ratio (relatively to the average local density) to keep a candidate point
params[
    'minHorizontalDensity'] = args.minHorizontalDensity  # the minimal density on horizontal axis to keep an interval in the final result
params[
    'maxDistToTheDiagonal'] = args.maxDistToTheDiagonal  # the maximal distance to the diagonal (inside a given interval) for a point to be taken into account in the horizontal density
params['kBest'] = args.kBest  # number of best coordinates to keep in creating points
params['margin'] = args.margin  # margin : min distance between neighbouring sentences
params['maxGapSize'] = args.maxGapSize  # max distance between two points to make a gap between two interval
params['diagBeam'] = args.diagBeam  # max distance to the diagonal
params['localDiagBeam'] = args.localDiagBeam  # max distance to the diagonal in the interval
params['reiterateFiltering'] = args.reiterateFiltering
params['useShelve'] = args.useShelve
params['wordAlignment'] = args.wordAlignment
params['chunkAlignment'] = args.chunkAlignment
params['outputFormats'] = args.outputFormats
params['outputDir'] = args.outputDir

# various low level parameters
print_log = True
show_plot_4_new_interval = False
min_sent_length_ratio = 0.2  # the minimal ratio between the shorter and the longer sentence to yield a candidate point
min_sent_length = 1  # the minimal sentence size to look for ngram
coeff_sent_len = 0.33  # balance between sentence based length
coeff_neighbour_sim = 0.6  # strength of the margin penalty
seg_min_length = 5  # min length for an aligned segment (in order to avoid oversegmentation)
use_encoder = False  # to compute the embeddings of sentence concatenations
max_group_size = 4
print_gap = False
params['verbose'] = True
embed_shelve = {}
xml_id_offset = 0
match_first_pre_anchors = True

################################################################
# initialization code

infinite = float('inf')
allowed_groups = []
print_plot = params['savePlot'] or params['showPlot']
only_one_2_one_pairing = False

allowed_groups = [(0, 1), (1, 0), (1, 1)]
if not only_one_2_one_pairing:
    for i in range(2, max_group_size + 1):
        allowed_groups.append((1, i))
        allowed_groups.append((i, 1))
    if params['noEmptyPair']:
        allowed_groups.remove((1, 0))
        allowed_groups.remove((0, 1))
    if not params['no2_2Group']:
        allowed_groups.append((2, 2))

if params['verbose']:
    print(f"Allowed groups : {allowed_groups}")

# to optimize parameters, temporarily save the embeddings in shelve or load embeds from the shelves
# N.B : embeds are normalized

log = None

# opening log and models if necessary
if print_log:
    log = open(os.path.join(output_dir, "ailign.log"), mode="a", encoding="utf8")
    now = datetime.now()
    # Formater la date et l'heure
    formatted_date = now.strftime("%d-%m-%Y, %H:%M:%S")
    log.write("\n"+formatted_date+"\nExecution of : "+" ".join(sys.argv)+"\n")

# conditionnaly import alternative models (main model is labse)
preprocessor = False
encoder = False

# open various pretrained models (https://www.sbert.net/docs/pretrained_models.html) including labse
# n.b.: some model are more adapted to translation comparison, other to paraphrasing
if params['embedModel'] == "laser":
    # import modules for laser
    from laserembeddings import Laser

    encoder = Laser()
elif params['embedModel'] == "sbert":
    # import modules for sbert
    from sentence_transformers import SentenceTransformer

    print("*** Loading sbert model", params['modelName'])
    encoder = SentenceTransformer(params['modelName'])
elif params['embedModel'] == "labse-keras":
    import tensorflow_hub as hub
    import tensorflow as tf
    import tensorflow_text as text  # Needed for loading universal-sentence-encoder-cmlm/multilingual-preprocess

    print("*** LABSE : Loading preprocessor")
    preprocessor = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
    print("*** LABSE : Loading model")
    encoder = hub.KerasLayer("https://tfhub.dev/google/LaBSE/2")

# init segmentation
segmenter = None

# parameter for sentence segmentation
if params['splitSent']:
    if params['useSentenceSegmenter']:
        from trankit import Pipeline

        if params['verbose']:
            print("Loading sentence segmenter from trankit")
        # names are defined here : https://trankit.readthedocs.io/en/latest/pkgnames.html
        names = {
            "en": "english",
            "de": "german",
            "es": "spanish",
            "fr": "french",
            "zh": "chinese",
            "ar": "arabic",
            "it": "italian"
        }
        try:
            segmenter = Pipeline(names[l1])
            segmenter.add(names[l2])
        except:
            print(
                f"Error while loading sentence segmenter from trankit. Check that you have defined a name for languages {l1} and {l2} (line 240)")
    else:
        # Rules that define sentence segmentation
        split_sent_regex = {
            'zh': r'(?<=[：，。？！”])',
            'ar': r'(?<=\.|۔)',
            'fr': r'(?<=[.!?;:])\s+(?=[A-Z«"])|(?<=[!?;:])',  # grimm Baudry
            'de': r'(?<=[.!?;:’“]) (?=[A-Z«"„])|(?<=[!?;:])|(?=[‘“])',  # grimm KHM 1857
            'grc': r'(?<=[?;:.!"»…])\s',
            'default': r'(?<=[?;:.!"»…]) (?=[A-Z])',
        }

# Rules that define a correct end of line, for line merging


merge_lines_regex = {
    'zh': r'[：，。？！”]\s*$',
    'fr': r'[?;:\.!"»…]\s*$',
    'ar': r'(\.|۔)\s*$'
}

########################################

# arc format is adapted to yasa input
arc_header = "\n<text>\n<divid='d1'>\n<pid='d1p1'>\n"
arc_footer = "</p>\n</div>\n</text>\n"

# ces format is another standard for segmented files
ces_header = """<?xml version="1.0" encoding="utf-8"?>
<cesAna>
<chunkList>
<chunk>
<par>
"""
ces_footer = """
</par>
</chunk>
</chunkList>
</cesAna>"""

# cesalign format is used to store alignment result
ces_align_header = f"""<?xml version="1.0" encoding="utf-8"?>

<cesAlign type="seg" version="1.6">

<ces_header version="2.3" mean_score="__mean_score__">
    <translations>
        <translation lang="{l1}" />
        <translation lang="{l2}" />
    </translations>
</ces_header>

<linkList>
    <linkGrp targType="seg">

"""
ces_align_footer = """
</linkGrp>
</linkList>

</cesAlign>
"""
# tmx is a common xml format to encode aligned file (for translation memories)
tmx_header = """
<?xml version="1.0" encoding="utf-8" ?>
<!DOCTYPE tmx SYSTEM "tmx14.dtd">
<tmx version="1.4">
  <header
    creationtool="AIlign"
    creationtoolversion="1.0"
    datatype="unknown"
    segtype="sentence"
    mean_score="__mean_score__"
  >
  </header>
  <body>
"""

tmx_footer = """
  </body>
</tmx>  
"""


def toXML(s):
    s = re.sub(r'&', '&amp;', s);
    s = re.sub(r'<', '&lt;', s);
    s = re.sub(r'>', '&gt;', s);
    return s


########################################################################### filtering points functions
# computation of local density
# local space may be centered, or before (for a point wich ends an interval)
# or after (for a point that begins an interval).
# max density is taken
def compute_local_density(i, j, points, I, J, sim_mat, delta_x, delta_y):
    coeff = J / I if params['sentRatio'] == 0 else params['sentRatio']
    local_space_size_before = 0
    nb_points_in_local_space_size_before = 0

    local_space_size_centered = 0
    nb_points_in_local_space_size_centered = 0

    local_space_size_after = 0
    nb_points_in_local_space_size_after = 0

    for X in range(max(0, i - 2 * delta_x), min(i + 2 * delta_x + 1, I)):
        for Y in range(int(max(0, j - (i - X) * coeff - delta_y)), int(min(j - (i - X) * coeff + delta_y + 1, J))):
            if X <= i:
                local_space_size_before += 1
                if (X, Y) in points.keys():
                    nb_points_in_local_space_size_before += sim_mat[X, Y]
            if X >= i:
                local_space_size_after += 1
                if (X, Y) in points.keys():
                    nb_points_in_local_space_size_after += sim_mat[X, Y]
            if max(0, i - delta_x) <= X < min(i + delta_x + 1, I):
                local_space_size_centered += 1
                if (X, Y) in points.keys():
                    nb_points_in_local_space_size_centered += sim_mat[X, Y]

    (densityBefore, densityAfter, densityCentered) = (0, 0, 0)
    if local_space_size_before:
        densityBefore = nb_points_in_local_space_size_before / local_space_size_before
    if local_space_size_after:
        densityAfter = nb_points_in_local_space_size_after / local_space_size_after
    if local_space_size_centered:
        densityCentered = nb_points_in_local_space_size_centered / local_space_size_centered
    return max(densityBefore, densityAfter, densityCentered)


# filtering points by eliminating every point in the center of a low density local area
# output : (points,filtered_x,filtered_y)
def filter_points(pre_anchors_x, points, I, J, average_density, sim_mat, delta_x, delta_y):
    # initialisation of filtered points
    filtered_x = []
    filtered_y = []
    nbDeleted = 0

    if params['veryVerbose']:
        print("Filtering ", len(points), "...")

    # computation of local density for each point
    points_key = sorted(list(points.keys()), key=lambda point: point[0])

    for point in points_key:
        (i, j) = point

        localDensity = compute_local_density(i, j, points, I, J, sim_mat, delta_x, delta_y)

        if params['veryVerbose']:
            print("i=", i, "j=", j, "Local density=", localDensity, "Average density=", average_density, "Ratio=",
                  round(localDensity / average_density, 2))

        # point is removed if density is not high enough
        if i not in pre_anchors_x and average_density > 0 and localDensity / average_density < params['minDensityRatio']:
            del (points[(i, j)])
            nbDeleted += 1

            # ~ x=[p[0] for p in points ]
            # ~ y=[p[1] for p in points]
            # ~ plt.axis([0,I,0,J])
            # ~ plt.title(str(i)+","+str(j)+'=> low density')
            # ~ plt.scatter(x,y,c="black",s=1)
            # ~ plt.scatter([i],[j],c="red",s=1)
            # ~ (i1,j1)=(i-delta_x,j-delta_x-delta_y)
            # ~ (i1,j2)=(i-delta_x,j-delta_x+delta_y)
            # ~ (i2,j3)=(i+delta_x,j+delta_x+delta_y)
            # ~ (i2,j4)=(i+delta_x,j+delta_x-delta_y)
            # ~ X=[i1,i1,i2,i2,i1]
            # ~ Y=[j1,j2,j3,j4,j1]
            # ~ plt.plot(X,Y,c="grey")
            # ~ plt.show()

        else:
            filtered_x.append(i)
            filtered_y.append(j)

    if params['verbose']:
        print(nbDeleted, "points have been removed!")

    return (points, filtered_x, filtered_y)


# removing points that are conflicting on the same column : only the point with the higher local density is kept
def resolving_conflicts(points, I, J, sim_mat):
    x2y = {}
    y2x = {}
    filtered_x = []
    filtered_y = []
    nbDeleted = 0
    points_key = list(points.keys())
    for point in points_key:
        (i, j) = point
        # conflict on x coordinate
        if i in x2y.keys():
            if x2y[i] != j:
                # for x coordinate, conflict between (i,j) and (i,x2y[i])
                # only the best point is kept
                density1 = compute_local_density(i, j, points, I, J, sim_mat, params['deltaX'], params['deltaY'])
                density2 = compute_local_density(i, x2y[i], points, I, J, sim_mat, params['deltaX'], params['deltaY'])
                nbDeleted += 1
                if density1 > density2:
                    if (i, x2y[i]) in points:
                        del (points[(i, x2y[i])])
                    x2y[i] = j
                else:
                    del (points[(i, j)])
                    continue
        else:
            x2y[i] = j

        if j in y2x.keys():
            if y2x[j] != i:
                # for x coordinate, conflict between (i,j) and (i,x2y[i])
                # only the best point is kept
                density1 = compute_local_density(i, j, points, I, J, sim_mat, params['deltaX'], params['deltaY'])
                density2 = compute_local_density(y2x[j], j, points, I, J, sim_mat, params['deltaX'], params['deltaY'])
                nbDeleted += 1
                if density1 < density2:
                    if (y2x[j], j) in points:
                        del (points[(y2x[j], j)])
                    y2x[j] = i
                else:
                    del (points[(i, j)])
        else:
            y2x[j] = i

    if params['verbose']:
        print(nbDeleted, "conflicting points have been removed!")

    points_key = list(points.keys())
    for point in points_key:
        (i, j) = point
        filtered_x.append(i)
        filtered_y.append(j)
    return (points, filtered_x, filtered_y)


########################################################################### ngram points functions

# ngram that contain only the same repeated character are not valid (e.g. blank spaces...)
def valid(ngram):
    return not re.match(r'(.)\1+', ngram)


# extract candidates points using ngram search
def computePointsFromNgrams(sents1, sents2):
    # extracting hash table that records all the ngrams for sents1
    len_sents1 = len(sents1)
    len_sents2 = len(sents2)
    sim_mat=np.array([0]*len_sents1*len_sents2)
    sim_mat.shape=len_sents1,len_sents2
    ngrams1 = []
    for i in range(len_sents1):
        ngrams1.append({})
        sent1 = sents1[i]
        for k in range(0, len(sent1) - params['ngram']):
            ngram = sent1[k:k + params['ngram']]
            if valid(ngram):
                if ngram not in ngrams1[i].keys():
                    ngrams1[i][ngram] = 0
                ngrams1[i][ngram] += 1

    # extracting hash table that records all the ngrams for sents2
    ngrams2 = []
    for j in range(len_sents2):
        sent2 = sents2[j]
        ngrams2.append({})
        for k in range(0, len(sent2) - params['ngram']):
            ngram = sent2[k:k + params['ngram']]
            if valid(ngram):
                if ngram not in ngrams2[j].keys():
                    ngrams2[j][ngram] = 0
                ngrams2[j][ngram] += 1

    # record the corresponding coordinate, sorted according to dice
    bestJ = {}
    bestI = {}

    # Using diagBeam param
    if params['diagBeam']:
        range2 = len_sents2 * params['diagBeam']
    else:
        range2 = len_sents2
    # dice computation for each point (i,j)
    for i in range(len_sents1):
        nb1 = max(1, len(sents1[i]) - params['ngram'] + 1)
        if params['verbose'] and i % 100 == 0:
            print("x =", i, "/", len_sents1)
        for J in range(range2):
            if params['diagBeam']:
                # when using fixed vertical width around diag, j must be computed as: int(i*len_sents2/len_sents1-range2/2)
                j = int(i * len_sents2 / len_sents1 - range2 / 2)
            else:
                j = J
            if j < 0:
                continue
            nb2 = max(1, len(sents2[j]) - params['ngram'] + 1)
            # length of sent1 and sent2 must be comparable
            if nb1 > min_sent_length and nb2 > min_sent_length and nb1 / nb2 >= min_sent_length_ratio and nb2 / nb1 >= min_sent_length_ratio:
                # computing the number of common ngrams (based on occurrences and not on type)
                nbCommon = 0
                for ngram in ngrams1[i].keys():
                    if ngram in ngrams2[j].keys():
                        nbCommon += min(ngrams1[i][ngram], ngrams2[j][ngram])
                dice = 2 * nbCommon / (nb1 + nb2)
                sim_mat[i,j]=dice
                # if dice is greater than the threshold, candidate point (i,j) is recorded
                if dice > params['diceThreshold']:
                    if not j in bestI.keys():
                        bestI[j] = []
                    if not i in bestJ.keys():
                        bestJ[i] = []
                    bestI[j].append((dice, i))
                    bestJ[i].append((dice, j))
    
    (points, x, y)=kBestPoints(bestI, bestJ)
    return (points, x, y, sim_mat)


def kBestPoints(bestI, bestJ):
    # building the point list taking, for each coordinate, the k best corresponding point
    x = []
    y = []
    points = {}  # points are recorded here as keys
    for i in bestJ.keys():
        # sorting the candidate according to sim
        bestJ[i] = sorted(bestJ[i], key=lambda x: x[0], reverse=True)
        if len(bestJ[i]) > 1:
            if (bestJ[i][0][0] - bestJ[i][1][0]) < params['margin']:
                if params['verbose']:
                    print("Filtering using margin criterion : ", bestJ[i][0][0], "-", bestJ[i][1][0], "<",
                          params['margin'])
                bestJ[i] = ()
            else:
                # only the k best are recorded
                bestJ[i] = [bestJ[i][l][1] for l in range(0, min(params['kBest'], len(bestJ[i])))]

    for j in bestI.keys():
        # sorting the candidate according to dice
        bestI[j] = sorted(bestI[j], key=lambda x: x[0], reverse=True)
        if len(bestI[j]) > 1:
            if (bestI[j][0][0] - bestI[j][1][0]) < params['margin']:
                if params['verbose']:
                    print("Filtering using margin criterion : ", bestI[j][0][0], "-", bestI[j][1][0], "<",
                          params['margin'])
                bestI[j] = ()
            else:
                # only the k best are recorded
                bestI[j] = [bestI[j][l][1] for l in range(0, min(params['kBest'], len(bestI[j])))]

    for i in bestJ.keys():
        for j in bestJ[i]:
            if j in bestI and i in bestI[j]:
                x.append(i)
                y.append(j)
                points[(i, j)] = 1
    return (points, x, y)


############################################################# LABSE points functions
# Function to normalize the embeddings by dividing them with their L2-norm
def normalization(embeds):
    norms = np.linalg.norm(embeds, 2, axis=1, keepdims=True)
    return embeds / norms


def computePointsWithEncoder(preprocessor, encoder, sents1, sents2):
    global embed_shelve
    points = {}  # points are recorded here as keys

    t0 = time.time()

    runEncoder = True
    # load from shelve in test mode (embeds are already computed)
    if params['useShelve']:
        embeds1 = []
        embeds2 = []
        runEncoder = False
        for sent in sents1:
            if sent in embed_shelve:
                embeds1.append(embed_shelve[sent])
            else:
                runEncoder = True
                break
        for sent in sents2:
            if sent in embed_shelve:
                embeds2.append(embed_shelve[sent])
            else:
                runEncoder = True
                break
    if runEncoder:
        if params['verbose']:
            print("Running Encoder...\n")

        embeds1 = computeEmbeds(preprocessor, encoder, params['embedModel'], sents1, l1)
        embeds2 = computeEmbeds(preprocessor, encoder, params['embedModel'], sents2, l2)

        t1 = time.time()
        if params['verbose']:
            print("\n1. Encoding -->", t1 - t0, "s.\n")

        # For semantic similarity tasks, apply l2 normalization to embeddings
        embeds1 = normalization(embeds1)
        embeds2 = normalization(embeds2)
        t2 = time.time()
        if params['verbose']:
            print("\n2. Normalization -->", t2 - t1, "s.\n")

    # saving normalized embeddings to shelve
    if params['useShelve'] and runEncoder:
        for i, sent in enumerate(sents1):
            embed_shelve[sent] = embeds1[i]
        for i, sent in enumerate(sents2):
            embed_shelve[sent] = embeds2[i]
        t2 = time.time()
        if params['verbose']:
            print("1-2. Loading embeddings from shelve -->", t2 - t0, "s.\n"),
    t3 = time.time()
    # similarity
    mat = np.matmul(embeds1, np.transpose(embeds2))

    t4 = time.time()
    if params['verbose']:
        print("\n3. Similarity matrix -->", t4 - t3, "s.\n"),

    # building the point list taking, for each coordinate, the k best corresponding point
    x = []
    y = []
    points = {}  # points are recorded here as keys

    # if the searchspace is reduced around the diagonal, compute the kBest point manually
    if params['diagBeam'] < 1:
        # record the corresponding coordinate
        bestJ = {}
        bestI = {}
        maxVertDistToTheDiagonal = int(len(sents2) * params['diagBeam'])
        for i in range(len(mat)):
            diagJ = int(i / len(mat) * len(mat[i]))
            infJ = max(0, diagJ - maxVertDistToTheDiagonal)
            supJ = min(diagJ + maxVertDistToTheDiagonal, len(mat[i]))
            for j in range(infJ, supJ):
                if mat[i][j] > params['cosThreshold']:
                    if i not in bestJ:
                        bestJ[i] = []
                    if j not in bestI:
                        bestI[j] = []
                    bestJ[i].append((mat[i][j], j))
                    bestI[j].append((mat[i][j], i))
        t5 = time.time()
        if params['verbose']:
            print("\n4. Extracting points -->", t5 - t4, "s.\n"),

        (points, x, y) = kBestPoints(bestI, bestJ)
        t6 = time.time()
        if params['verbose']:
            print("\n5. Filtering k best vertically and horizontally -->", t6 - t5, "s.\n"),

    # use numpy argpartition for kBest extraction
    else:
        k = params['kBest']
        # for k=1 we extract the 2 best, in order to apply the margin criterion
        if k == 1:
            k = 2
        k = min(k, len(mat[0]))
        # using argpartition allow to extract quickly the k-best col for each line
        ind_by_line = np.argpartition(mat, -k, axis=1)[:, -k:]
        sim_by_line = np.take_along_axis(mat, ind_by_line, axis=1)
        bestJ = [list(zip(sim_by_line[i], ind_by_line[i])) for i in range(len(sim_by_line))]

        for i in range(len(bestJ)):
            bestJ[i].sort(key=lambda x: x[0], reverse=True)
            if (bestJ[i][0][0] - bestJ[i][1][0]) < params['margin']:
                if params['veryVerbose']:
                    print("Filtering using margin criterion : ", bestJ[i][0][0], "-", bestJ[i][1][0], "<",
                          params['margin'])
                bestJ[i] = []
            # once margin criterion has been applied, apply the threshold
            bestJ[i] = [pair for pair in bestJ[i] if pair[0] > params['cosThreshold']]
            # if kBest==1, crop the candidate list
            if params['kBest'] == 1 and len(bestJ[i]) > 1:
                bestJ[i] = bestJ[i][0:1]

        ind_by_col = np.argpartition(mat, -k, axis=0)[-k:, :]
        sim_by_col = np.take_along_axis(mat, ind_by_col, axis=0)
        ind_by_col = ind_by_col.swapaxes(1, 0)
        sim_by_col = sim_by_col.swapaxes(1, 0)

        bestI = [list(zip(sim_by_col[i], ind_by_col[i])) for i in range(len(ind_by_col))]
        for j in range(len(bestI)):
            bestI[j].sort(key=lambda x: x[0], reverse=True)
            if (bestI[j][0][0] - bestI[j][1][0]) < params['margin']:
                if params['veryVerbose']:
                    print("Filtering using margin criterion : ", bestI[j][0][0], "-", bestI[j][1][0], "<",
                          params['margin'])
                bestI[j] = []
            # once margin criterion has been applied, apply the threshold
            bestI[j] = [pair for pair in bestI[j] if pair[0] > params['cosThreshold']]
            # if kBest==1, crop the candidate list
            if params['kBest'] == 1 and len(bestI[j]) > 1:
                bestI[j] = bestI[j][0:1]

        # adding points
        for i in range(len(bestJ)):
            for n in range(len(bestJ[i])):
                j = bestJ[i][n][1]
                for m in range(len(bestI[j])):
                    if i == bestI[j][m][1]:
                        x.append(i)
                        y.append(j)
                        points[(i, j)] = 1
                        break

        t5 = time.time()
        if params['verbose']:
            print("\n4-5. Extracting and filtering k best vertically and horizontally -->", t5 - t4, "s.\n"),

    return (points, x, y, mat, embeds1, embeds2)


############################################################# LASER and BERT points functions


# return the normalized embeddings for a given encoder and a sentence list
def computeEmbeds(preprocessor, encoder, embed_model, sents, language=""):
    if embed_model == "laser":
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

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Function to compute the similarity matrix and identify similar sentences
# ~ def computePoints(tokenizer, model, embed_model, sents1, sents2):
    # ~ points = {}  # Dictionary to store the indices of similar sentences
    # ~ t0 = time.time()  # Record the starting time for performance measurement

    # ~ if embed_model == "bert":
        # ~ if params['verbose']:
            # ~ print("Running Encoder...\n"),
        # ~ # Tokenize the input sentences using the BERT tokenizer
        # ~ inputs1 = tokenizer(sents1, return_tensors='pt', padding=True, truncation=True)
        # ~ inputs2 = tokenizer(sents2, return_tensors='pt', padding=True, truncation=True)
        # ~ t1 = time.time()
        # ~ if params['verbose']:
            # ~ print("1. Encoding -->", t1 - t0, "s.\n")  # Print the time taken for encoding

        # ~ # Pass the tokenized inputs through the BERT model
        # ~ outputs1 = model(**inputs1)
        # ~ outputs2 = model(**inputs2)

        # ~ # Extract the embeddings from the BERT model's output and convert them to numpy arrays
        # ~ embeds1 = outputs1.last_hidden_state[:, 0, :]
        # ~ embeds2 = outputs2.last_hidden_state[:, 0, :]

        # ~ # Normalize the embeddings using the normalization function
        # ~ embeds1 = normalization(embeds1.detach().numpy())
        # ~ embeds2 = normalization(embeds2.detach().numpy())
        # ~ t2 = time.time()
        # ~ if params['verbose']:
            # ~ print("2. Normalization-->", t2 - t1, "s.\n")  # Print the time taken for normalization

    # ~ elif embed_model == "laser":
        # ~ # Use the Laser model to embed the sentences in different languages
        # ~ embeds1 = laser.embed_sentences(sents1, lang='fr')
        # ~ embeds2 = laser.embed_sentences(sents2, lang='en')
        # ~ t1 = time.time()
        # ~ if params['verbose']:
            # ~ print("1. Encoding -->", t1 - t0, "s.\n")  # Print the time taken for encoding

        # ~ # Normalize the embeddings using the normalization function
        # ~ embeds1 = normalization(embeds1)
        # ~ embeds2 = normalization(embeds2)
        # ~ t2 = time.time()
        # ~ if params['verbose']:
            # ~ print("2. Normalization-->", t2 - t1, "s.\n")  # Print the time taken for normalization

    # ~ # Compute the similarity matrix between the embeddings of the two sets of sentences
    # ~ mat = np.matmul(embeds1, embeds2.T)
    # ~ t3 = time.time()
    # ~ if params['verbose']:
        # ~ print("3. Similarity matrix -->", t3 - t2, "s.\n")  # Print the time taken for computing the similarity matrix

    # ~ x = []  # List to store the indices of similar sentences from sents1
    # ~ y = []  # List to store the indices of similar sentences from sents2
    # ~ bestJ = {}  # Dictionary to store the best match index for each sentence in sents1
    # ~ bestI = {}  # Dictionary to store the best match index for each sentence in sents2

    # ~ # Find the best match index for each sentence in sents1
    # ~ for i in range(len(mat)):
        # ~ m = 0
        # ~ for j in range(len(mat[i])):
            # ~ if mat[i][j] > m:
                # ~ m = mat[i][j]
                # ~ bestJ[i] = j

    # ~ # Find the best match index for each sentence in sents2
    # ~ for j in range(len(mat[0])):
        # ~ m = 0
        # ~ for i in range(len(mat)):
            # ~ if mat[i][j] > m:
                # ~ m = mat[i][j]
                # ~ bestI[j] = i
    # ~ t4 = time.time()
    # ~ if params['verbose']:
        # ~ print("4. Extracting best point according to horizontal and vertical axis-->", t4 - t3,
              # ~ "s.\n")  # Print the time taken for computing the similarity matrix

    # ~ # Identify the similar sentence pairs based on the best match indices and similarity threshold
    # ~ for i in range(len(mat)):
        # ~ j = bestJ[i]
        # ~ if bestI.get(j) == i and mat[i][j] >= params['cosThreshold']:
            # ~ x.append(i)
            # ~ y.append(j)
            # ~ points[(i, j)] = 1  # Store the indices of similar sentences in the points dictionary
    # ~ t5 = time.time()
    # ~ if params['verbose']:
        # ~ print("5. Filtering best points that exceed the threshold -->", t5 - t4,
              # ~ "s.\n")  # Print the time taken for computing the similarity matrix

    # ~ return points, x, y, mat, embeds1, embeds2


######################################################################### reading / writing files
# reading input file

def read_input_file(input_dir, inputFile, input_format, column=0, language="fr"):
    """Reads an input file and returns a list of sentences.

      Args:
        input_dir: The directory containing the input file.
        inputFile: The name of the input file.
        input_format: The format of the input file.
        column: The column number of the input file that contains the text.
        language: The language of the input file.

      Returns:
        sents: a list of sentences.
        id_sents: the list of sentence ids (build upon segment ids)
        len_sents: the number of sentences
        seg2sents: a list of list of integer, that gives the 1-n correspondence
            between an original segment number and the list of final sentences
            - if splitSent, for one segment, we may have more than one sentences
            - if mergeSent, more than one segment may correspond to the same merged sentence
    """
    global seg_min_length

    segs = []
    id_segs = []
    len_sents = 0
    seg2sents = []
    nb_chars = 0
    pre_anchors = [] # in XML format, it is possible to define anchors of prealignment

    try:
        input_file_path = os.path.join(input_dir, inputFile) if input_dir else inputFile
        f = open(input_file_path, encoding='utf8')
    except Exception as e :
        print("Error: a problem occurred while opening", inputFile, e)
        sys.exit()

    # Reading according to input_format
    if input_format == "arc" or input_format == "ces":
        for line in f:
            line = line.strip()
            m = re.search(r'<s\b[^>]+id="([^"]*)"', line)
            if m and m.group(1):
                id_segs.append(m.group(1))
            else:
                id_segs.append("s" + str(lenSegs))
            segs.append(line)
            lenSegs += 1
            nb_chars += len(line)


    # The json input contains a sentences property, which is a list sentences, which are list of tokens
    # Each token is a list of conll features, col1->form, col9=blank_space
    elif input_format == "json":
        content = f.read()
        jsonObj = json.loads(content)
        segs = [
            "".join([tok[1] + tok[9] for tok in sent if len(tok) >= 10]) for sent in jsonObj["sentences"]
        ]
        for seg in segs:
            nb_chars += len(seg)
        id_segs = [str(i) for i in list(range(0, len(segs) + 1))]

    # the tsv format is an already aligned format. Sentence are extracted from a specific column
    elif input_format == "tsv":
        segs = []
        for line in f:
            alignedSegs = re.split("\t", line)
            segs.append(alignedSegs[column])
            nb_chars += len(alignedSegs[column])
        id_segs = [str(i) for i in list(range(1, len(segs) + 1))]

    # in xml-conll, the conll sentences are encoded between <s></s> markup
    elif input_format == "xml-conll":
        content = f.read()
        try:
            xml_root = ET.fromstring(content)
        except :
            print("non conform XML :", os.path.join(input_dir, inputFile))
            print(sys.exc_info()[0])
            # error_log.write("non conform XML :",os.path.join(input_dir, inputFile),"\n")
            sys.exit()

        for s_elt in xml_root.findall('.//s'):
            s = "".join(s_elt.itertext())
            # suppression des tabulations
            s = re.sub(r"\t","",s)
            toks = []
            for line in re.split(r"\n", s):
                cols = re.split("\t", line)
                if len(cols) == 10:
                    toks.append(cols[1])
            seg = " ".join(toks)
            segs.append(seg)
            nb_chars += len(seg)

            if s_elt.attrib["id"]:
                id_segs.append(s_elt.attrib["id"])
            elif s_elt.attrib["xml:id"]:
                id_segs.append(s_elt.attrib["xml:id"])
            else:
                id_segs.append(str(len(segs)))

    # In XML format, the sentences are extracted using the text content of
    # the elements that are defined by xmlGuide (a list of tag or simple xpath expressions)
    elif input_format == "xml":
        content = f.read()
        content = re.sub(r'xmlns="[^"]*"', "", content)
        try:
            xml_root = ET.fromstring(content)
        except Exception as err :
            print("non conform XML :", os.path.join(input_dir, inputFile))
            print(err)
            # error_log.write("non conform XML :",os.path.join(input_dir, inputFile),"\n")
            sys.exit()
        segs = []
        # text element is default anchor 
        anchor_xpath= ".//" +params['anchorTag'] if params['anchorTag'] else ".//text"
        for prealigned_elt in xml_root.findall(anchor_xpath):
            if  params['anchorTag']:
                # when an anchor or prealignment is found, feed the preAnchors list
                pre_anchors.append(len(segs))
            # special case where xmlGuide=anchorTag
            if params['anchorTag'] and params['anchorTag'] in params['xmlGuide']:
                prealigned_elts=[prealigned_elt]
            else:
                xpath = '|'.join([".//" + tag for tag in params['xmlGuide'] if tag!=""])
                prealigned_elts=prealigned_elt.findall(xpath)

            for elt in prealigned_elts:
                content = "".join(elt.itertext())
                content = re.sub(r"[\n\t]", " ", content)
                segs.append(content)
                params['verbose'] and print("Adding sentence n°",len(segs))

                nb_chars += len(content)
                # recording id in id_segs
                if 'id' in elt.attrib:
                    id_segs.append(elt.attrib["id"])
                elif "xml:id" in elt.attrib:
                    id_segs.append(elt.attrib["xml:id"])
                else:
                    id_segs.append(str(len(segs)))

    # Default format: one sentence per line
    else:
        print("Warning : default format TXT")
        for line in f:
            line = line.strip()
            line = re.sub(r'\x0A|\x0D', '', line)
            nb_chars += len(line)
            segs.append(line)
        id_segs = [str(i) for i in list(range(1, len(segs) + 1))]

    # Here, the lines that corresponds to the same sentences may be merged
    # The corresponding sentence id will result in the concatenation of initial segment id
    if params['mergeLines']:
        if len(pre_anchors)>0:
            print("******* Attention avec le paramètre 'mergeLines' on ne peut prendre en compte les balises de préalignement")
            pre_anchors.clear()
        if params['verbose']:
            print("Line merging for ", language)
        sents = []
        id_sents = []
        num_sents = 0
        current_sent = []
        current_ids = []
        for (i, seg) in enumerate(segs):
            seg2sents.append([num_sents])
            current_ids.append(id_segs[i])
            current_sent.append(seg)
            # merging when the sentence ends with a separator
            if re.search(merge_lines_regex[language], seg) or seg.upper() == seg:
                id_sents.append("-".join(current_ids))
                sents.append(" ".join(current_sent))
                current_sent = []
                current_ids = []
                num_sents += 1
        if len(current_ids) > 0:
            sents.append(" ".join(current_sent))
            id_sents.append("-".join(current_ids))

    # here, segments can be split in smaller pieces
    elif params['splitSent']:
        if params['verbose']:
            print("Sentence segmentation for ", language)
        sents = []
        id_sents = []
        new_pre_anchors=[]
        if params['useSentenceSegmenter']:
            segmenter.set_active(names[language])
        for (i, seg) in enumerate(segs):
            if i in pre_anchors:
                new_pre_anchors.append(len(sents))
            # use trankit for sentence segmentation
            if params['useSentenceSegmenter']:
                print("segmentation de ", seg)
                sentences = segmenter.ssplit(seg)['sentences']
                some_sents = [sent['text'] for sent in sentences]
            # or use a set of regex declared in splitSent
            else:
                if params["splitSentRegex"]:
                    regex=params["splitSentRegex"]
                elif language in split_sent_regex:
                    regex = split_sent_regex[language]
                else:
                    regex = split_sent_regex["default"]
                some_sents = re.split(regex, seg)
            
            last_sent = ""
            new_sents = []
            # the splitted segment that are too small (< seg_min_length)
            # are grouped with the follower
            for sent in some_sents:
                if not re.match(r'^\s*$',sent):
                    if len(last_sent + sent) > seg_min_length:
                        new_sents.append(last_sent + " " + sent)
                        last_sent = ""
                    else:
                        if last_sent == "":
                            last_sent = sent
                        else:
                            last_sent += " " + sent
            if last_sent:
                new_sents.append(last_sent)

            seg2sents.append(list(range(len(sents), len(sents) + len(new_sents))))
            new_ids = [id_segs[i]]
            if len(new_sents) > 1:
                new_ids = [id_segs[i] + "_" + str(j) for j in range(len(new_sents))]
            sents.extend(new_sents)
            id_sents.extend(new_ids)
        # updating pre_anchors
        pre_anchors=new_pre_anchors

    # keeping the same segments as in the input
    else:
        sents = segs
        id_sents = id_segs
        seg2sents = [[j] for j in range(len(sents))]

    len_sents = len(sents)
    if params['verbose']:
        print(len_sents, "sentences for ", language)
        if params['veryVerbose']:
            print("\n".join(sents))

    f.close()

    if params['writeSegmentedInput']:
        input_file_pathSeg = re.sub(r"(.*)([._]\w+)[.]\w+$", r"\1.seg\2.txt", input_file_path)
        if input_file_pathSeg==input_file_path:
            input_file_pathSeg=input_file_path+".seg"
        seg_file = open(input_file_pathSeg, mode="w", encoding="utf8")
        seg_file.write("\n".join([ (f"<anchor/> {i}: " if i in pre_anchors else f"{i}: ")+sent for i,sent in enumerate(sents)]))
        seg_file.close()

    return (sents, id_sents, len_sents, seg2sents, nb_chars, pre_anchors)


# write only alignable intervals of l1 or l2 file
def write_alignable(sents, id_sents, intervals, index, output_dir, output_file, output_format):
    """
    Arguments :
        sents : List(str) : the sentence list
        id_sents : List(str) : the corresponding sentence ids
        intervals : List(List(int)) : the list of pairs [i..j] that defines corresponding intervals. The second axe is the language : 0 for l1, 1 for l2
        index : 0 or 1 for l1 or l2
        output_dir : str : the path of output dir
        output_file : str : the name of output file
        output_format : str : "ces" or "arc" or "txt"

    No return value, but the file output_file is written on the disk
    """

    try:
        output = open(os.path.join(output_dir, output_file), mode="w", encoding="utf8")
        # output header
        if output_format == "ces":
            output.write(ces_header)
        elif output_format == "arc":
            output.write(arc_header)

        # output sentences
        for interval in intervals:
            i1 = interval[0][index]
            i2 = interval[1][index]

            for i in range(i1, i2 + 1):
                if output_format == "ces" or output_format == "arc":
                    if input_format == "ces" or output_format == "arc":
                        id_sent = id_sents[i]
                    else:
                        id_sent = str(i + 1)
                    output.write("<s id=\"" + id_sent + "\">\n" + toXML(sents[i]) + "\n</s>\n")
                else:
                    output.write(sents[i] + "\n")

        # output footer
        if output_format == "ces":
            output.write(ces_footer)
        elif output_format == "arc":
            output.write(arc_footer)

        output.close()
    except:
        print("Error: a problem occurred while writing", output_file)
        sys.exit()


# write aligned points
# if the anchor parameter is true then filtered_x and filtered_y are list of int
# if not, they are list of list of int (the grouped coordinate)
def write_aligned_points(l1, l2, sents1, id_sents1, sents2, id_sents2, filtered_x, filtered_y, output_dir, output_file,
                         output_format, anchor, print_ids=False, mean_score=0, file1="", file2=""):
    """
    Arguments :
        sents1 : List(str) : the L1 sentence list
        id_sents1 : List(str) : the corresponding sentence ids
        sents2 : List(str) : the L2 sentence list
        id_sents2 : List(str) : the corresponding sentence ids
        filtered_x : List(List(int)) OR List(int) if anchor=True
            if anchor == false : the X coordinates of groups in L1 (ex. :[[0],[1,2],[3],[4,5,6]])
            if anchor == true : the X coordinates of points in L1 (ex. [0, 2, 3, 5])
        filtered_y : List(List(int)) OR List(int) if anchor=True
            if anchor == false : the Y coordinates of groups in L2 (ex. :[[0,1],[2],[3],[4,5]])
            if anchor == true : the Y coordinates of points in L2 (ex. [1, 2, 3, 5])
        output_dir : str : the path of output dir
        output_file : str : the name of output file
        output_format : str : "tmx" or "ces" or "ids" or "txt"

    No return value, but the file output_file is written on the disk
    """

    global tmx_header, ces_align_header

    # ~ try:
    if output_format == "txt2":
        output_file1 = output_file.replace(".txt2", "." + l1 + ".txt")
        output1 = open(os.path.join(output_dir, output_file1), mode="w", encoding="utf8")
        output_file2 = output_file.replace(".txt2", "." + l2 + ".txt")
        output2 = open(os.path.join(output_dir, output_file2), mode="w", encoding="utf8")
    else:
        if output_file[-4:] == "tsv2":
            output_file = output_file[:-1]
        output = open(os.path.join(output_dir, output_file), mode="w", encoding="utf8")
    # ~ output2=open(os.path.join(output_dir,output_file+".txt"),mode="w",encoding="utf8")
    # output header
    if output_format == "ces":
        output.write(re.sub(r'__mean_score__', f"{mean_score:.4f}", ces_align_header))
    elif output_format == "tmx":
        output.write(re.sub(r'__mean_score__', f"{mean_score:.4f}", tmx_header))
    elif output_format == "txt":
        output.write(f"Mean similarity:{mean_score}\n")
    elif output_format == "txt2":
        output1.write(f"Mean similarity:{mean_score}\n")
        output2.write(f"Mean similarity:{mean_score}\n")
    elif output_format == "tsv2":
        # ~ m=re.match('(.*)[.][^.]+[.][^.]+$',output_file)
        # ~ name=m.group(1)
        name1 = os.path.basename(file1)
        name2 = os.path.basename(file2)
        output.write(f"source={l1}/{collection_name}/{name1}	target={l2}/{collection_name}/{name2}\n\n")

    # output sentences
    for i in range(len(filtered_x)):
        if anchor:
            x = [filtered_x[i]]
            y = [filtered_y[i]]
        else:
            x = filtered_x[i]
            y = filtered_y[i]
        if output_format == "ces":
            if input_format == "ces" or input_format == "arc":
                id_sent1 = " ".join([id_sents1[x[j]] for j in range(len(x))])
                id_sent2 = " ".join([id_sents2[y[j]] for j in range(len(y))])
            else:
                id_sent1 = " ".join([str(x[j] + 1) for j in range(len(x))])
                id_sent2 = " ".join([str(y[j] + 1) for j in range(len(y))])
            output.write(f"\t\t<link xtargets=\"{id_sent1} ; {id_sent2}\"/>\n")
        elif output_format == "ids" or output_format == "tsv2":
            if input_format == "ces" or input_format == "arc":
                id_sent1 = " ".join([id_sents1[x[j]] for j in range(len(x))])
                id_sent2 = " ".join([id_sents2[y[j]] for j in range(len(y))])
            else:
                id_sent1 = " ".join([str(x[j] + 1) for j in range(len(x))])
                id_sent2 = " ".join([str(y[j] + 1) for j in range(len(y))])
                output.write(f"{id_sent1}\t{id_sent2}\n")
        elif output_format == "tmx":
            srcSegs = "".join(["\t\t<seg>" + toXML(sents1[x[j]]) + "</seg>\n" for j in range(len(x))])
            tgtSegs = "".join(["\t\t<seg>" + toXML(sents2[y[j]]) + "</seg>\n" for j in range(len(y))])

            output.write(f"<tu>\n")
            output.write(f"\t<tuv xml:lang=\"{l1}\">\n{srcSegs}\t</tuv>\n")
            output.write(f"\t<tuv xml:lang=\"{l2}\">\n{tgtSegs}\t</tuv>\n")
            output.write(f"</tu>\n")
        elif output_format == "txt":
            ids1 = "[" + " ".join([str(x[j]) for j in range(len(x))]) + "] " if print_ids else ""
            sent1 = ids1 + " ".join([sents1[x[j]] for j in range(len(x))])
            ids2 = "[" + " ".join([str(y[j]) for j in range(len(y))]) + "] " if print_ids else ""
            sent2 = ids2 + " ".join([sents2[y[j]] for j in range(len(y))])
            output.write(sent1 + "\n" + sent2 + "\n\n")
        elif output_format == "txt2":
            sent1 = " ".join(["[" + str(x[j]) + "] " + sents1[x[j]] for j in range(len(x))])
            output1.write(sent1 + "\n")
            sent2 = " ".join(["[" + str(y[j]) + "] " + sents2[y[j]] for j in range(len(y))])
            output2.write(sent2 + "\n")
        elif output_format == "tsv":
            ids1 = "[" + " ".join([str(x[j]) for j in range(len(x))]) + "] " if print_ids else ""
            sent1 = " ".join([sents1[x[j]] for j in range(len(x))])
            ids2 = "[" + " ".join([str(y[j]) for j in range(len(y))]) + "] " if print_ids else ""
            sent2 = " ".join([sents2[y[j]] for j in range(len(y))])
            output.write(f"{ids1}{sent1}\t{ids2}{sent2}\n")
        elif output_format == "bertalign":
            ids1 = "[" + ",".join([str(x[j]) for j in range(len(x))]) + "]"
            ids2 = "[" + ",".join([str(y[j]) for j in range(len(y))]) + "]"
            output.write(f"{ids1}:{ids2}\n")
        else:
            # default is TSV with no ID
            ids1 = "[" + " ".join([str(x[j]) for j in range(len(x))]) + "] " if print_ids else ""
            sent1 = ids1 + " ".join([sents1[x[j]] for j in range(len(x))])
            ids2 = "[" + " ".join([str(y[j]) for j in range(len(y))]) + "] " if print_ids else ""
            sent2 = ids2 + " ".join([sents2[y[j]] for j in range(len(y))])
            output.write(sent1 + "\t" + sent2 + "\n")

    # output footer
    if output_format == "ces":
        output.write(ces_align_footer)
    elif output_format == "tmx":
        output.write(tmx_footer)

    if output_format == "txt2":
        output1.close()
        output2.close()
    else:
        output.close()


# Writing anchors in xml files
def add_anchor_in_output(input_dir, input_file1, input_file2, file_id1, file_id2, x, y, output_dir, direction):
    """Adds anchors in input xml files

      Args:
        input_dir: The directory containing the input file.
        file1: The name of the file1.
        file2: The name of the file2.
        x: the source coordinates
        y: the corresponding target coordinates

      Returns:
        write files output_dir/file1 and output_dir/file2 (or input_dir/file1 and input_dir/file2 if output_dir is empty)
        adding anchors
    """

    # opening files
    try:
        input_file_path1 = os.path.join(input_dir, input_file1) if input_dir else input_file1
        f1 = open(input_file_path1, encoding='utf8')
    except:
        print("Error: a problem occurred while opening", input_file_path1)
        sys.exit()

    content1 = f1.read()
    content1 = re.sub(r'xmlns="[^"]*"|encoding="UTF-?8"', "", content1)
    f1.close()
    xml_root1 = etree.fromstring(content1)
    try:
        xml_root1 = etree.fromstring(content1)
        # ~ xml_root1 = ET.ElementTree(ET.fromstring(content1))
    except:
        print("non conform XML :", input_file_path1)
        sys.exit()

    try:
        input_file_path2 = os.path.join(input_dir, input_file2) if input_dir else input_file2
        f2 = open(input_file_path2, encoding='utf8')
    except:
        print("Error: a problem occurred while opening", input_file1)
        sys.exit()

    content2 = f2.read()
    content2 = re.sub(r'xmlns="[^"]*"|encoding=.UTF-?8.', "", content2)
    f2.close()
    try:
        xml_root2 = etree.fromstring(content2)
        # ~ xml_root2 = ET.ElementTree(ET.fromstring(content2))
    except:
        print("non conform XML :", input_file_path2)
        sys.exit()

    segs = []
    xpath = '|'.join(['.//' + tag for tag in params['xmlGuide']])
    # ~ sents1= xml_root1.findall(xpath)
    # ~ sents2= xml_root2.findall(xpath)
    sents1 = xml_root1.xpath(xpath)
    sents2 = xml_root2.xpath(xpath)

    for i in range(len(x)):
        if len(x[i]) > 0 and len(y[i]) > 0:
            xi = x[i][0]
            yi = y[i][0]
            corresp2_values = " ".join("#" + file_id1 + str(xi + xml_id_offset) for xi in x[i])

            if direction == "1<->2" or direction == "2->1":
                anchor1 = etree.Element("anchor")
                anchor1.set("{http://www.w3.org/XML/1998/namespace}id", file_id1 + str(xi + xml_id_offset))
                anchor1.set("corresp", "#" + file_id2 + str(yi + xml_id_offset))
                prev = sents1[xi].getprevious()
                if prev is not None:
                    prev.addnext(anchor1)
                else:
                    parent = sents1[xi].getparent()
                    parent.insert(0, anchor1)

            if direction == "1<->2" or direction == "1->2":
                anchor2 = etree.Element("anchor")
                anchor2.set("{http://www.w3.org/XML/1998/namespace}id", file_id2 + str(yi + xml_id_offset))
                anchor2.set("corresp", corresp2_values)
                prev = sents2[yi].getprevious()
                if prev is not None:
                    prev.addnext(anchor2)
                else:
                    parent = sents2[yi].getparent()
                    parent.insert(0, anchor2)
        elif len(y[i]) > 0:
            yi = y[i][0]
            if direction == "1<->2" or direction == "1->2":
                anchor2 = etree.Element("anchor")
                anchor2.set("{http://www.w3.org/XML/1998/namespace}id", file_id2 + str(yi + xml_id_offset))
                prev = sents2[yi].getprevious()
                if prev is not None:
                    prev.addnext(anchor2)
                else:
                    parent = sents2[yi].getparent()
                    parent.insert(0, anchor2)

    # add anchor before each source sentence
    if direction == '1->2':
        for i in range(len(sents1)):
            sent = sents1[i]
            anchor1 = etree.Element("anchor")
            anchor1.set("{http://www.w3.org/XML/1998/namespace}id", file_id1 + str(i + xml_id_offset))
            prev = sents1[i].getprevious()
            if prev is not None:
                prev.addnext(anchor1)
            else:
                parent = sents1[i].getparent()
                parent.insert(0, anchor1)
    elif direction == '2->1':
        for i in range(len(sents2)):
            sent = sents2[i]
            anchor2 = etree.Element("anchor")
            anchor2.set("{http://www.w3.org/XML/1998/namespace}id", file_id2 + str(i + xml_id_offset))
            prev = sents2[i].getprevious()
            if prev is not None:
                prev.addnext(anchor2)
            else:
                parent = sents2[i].getparent()
                parent.insert(0, anchor2)

    # writing output files
    if output_dir == "" or output_dir == None:
        output_file_path1 = input_file_path1
        output_file_path2 = input_file_path2
    else:
        output_file_path1 = os.path.join(output_dir, os.path.basename(input_file1))
        output_file_path2 = os.path.join(output_dir, os.path.basename(input_file2))

    try:
        tree1 = etree.ElementTree(xml_root1)
        print(f"Writing {output_file_path1}")
        tree1.write(output_file_path1, encoding='utf-8', pretty_print=True)
    except:
        print("Error: a problem occurred while writing", output_file_path1)
        sys.exit()

    try:
        tree2 = etree.ElementTree(xml_root2)
        print(f"Writing {output_file_path2}")
        tree2.write(output_file_path2, encoding='utf-8', pretty_print=True)
    except:
        print("Error: a problem occurred while writing", output_file_path2)
        sys.exit()


def extract_anchor_points(pre_anchors_x, pre_anchors_y, points, x, y, sents1, sents2, len_sents1, len_sents2, sim_mat):
    
    anchor_points = dict.copy(points)

    # =====> STEP 6 : compute average local density around selected points
    t5 = time.time()

    points_key = list(anchor_points.keys())

    if len(points_key) == 0:
        print("No anchor points !!!")
        beginInt = (-1, -1)
        lastI = len_sents1 - 1
        lastJ = len_sents2 - 1
        interval_length_sent1 = lastI - beginInt[0] + 1
        interval_length_sent2 = lastJ - beginInt[1] + 1
        for n in range(0, lastI + 1):
            interval_length_char1 += len(sents1[n])
        for n in range(0, lastJ + 1):
            interval_length_char2 += len(sents2[n])

    else:
        tot_density = 0
        for point in points_key:
            (x2, y2) = point
            tot_density += compute_local_density(x2, y2, anchor_points, len_sents1, len_sents2, sim_mat,
                                                 params['deltaX'], params['deltaY'])

        average_density = tot_density / float(len(points_key))

        t6 = time.time()
        if params['verbose']:
            print("\n6. Computing average density-->", t6 - t5, "s.\n"),

        # =====> STEP 7 : filtering out low density points

        (anchor_points, filtered_x, filtered_y) = filter_points(pre_anchors_x, anchor_points, len_sents1, len_sents2, 
                                                                average_density, sim_mat, params['deltaX'], params['deltaY'])
        (anchor_points, filtered_x, filtered_y) = resolving_conflicts(anchor_points, len_sents1, len_sents2, sim_mat)

        if params['reiterateFiltering']:
            (anchor_points, filtered_x, filtered_y) = filter_points(pre_anchors_x, anchor_points, 
                                                                    len_sents1, len_sents2,
                                                                    average_density * 2, sim_mat,
                                                                    int(params['deltaX'] / 2),
                                                                    int(params['deltaY'] / 2))

        t7 = time.time()
        if params['verbose']:
            print("\n7. Removing low density points-->", t7 - t6, "s.\n"),

        # ~ x=[point[0] for point in points]
        # ~ y=[point[1] for point in points]
        # ~ plt.axis([1,len_sents1,1,len_sents2])
        # ~ plt.title(output_file_name+'.txt - filtered')
        # ~ plt.scatter(x,y,c="red",s=1)
        # ~ plt.show()

        # =====> STEP 8 : finding aligned intervals

        beginInt = (-1, -1)
        # adding last point as an anchor
        filtered_x.append(len_sents1 - 1)
        filtered_y.append(len_sents2 - 1)
        lastI = 0
        lastJ = 0
        intervals = []  # the array of pairs (beginInt,endInt) where beginInt and endInd are two points that define the interval
        nb_in_interval = 0

        (interval_length_sent1, interval_length_sent2, interval_length_char1, interval_length_char2) = (0, 0, 0, 0)

        # adding pre_anchors as intervals to drive alignment
        if len(pre_anchors_x)>0:
            for x,y in zip(pre_anchors_x,pre_anchors_y):
                endInt=(x,y)
                if params['verbose']:
                    print ("Adding interval according to pre anchors ",(beginInt,endInt))
                intervals.append((beginInt,endInt))
                interval_length_sent1 += endInt[0] - beginInt[0] + 1
                interval_length_sent2 += endInt[1] - beginInt[1] + 1
                beginInt=endInt
            lastI = len_sents1 - 1
            lastJ = len_sents2 - 1
        elif params['detectIntervals']:
            coeff = 1 if params['sentRatio'] == 0 else params['sentRatio']
            for num in range(0, len(filtered_x)):
                (i, j) = (filtered_x[num], filtered_y[num])
                
                localDensity = compute_local_density(i, j, anchor_points, len_sents1, len_sents2, sim_mat,
                                                     params['deltaX'], params['deltaY'])
                density_ratio = 0
                if average_density > 0:
                    density_ratio = localDensity / average_density
                # computation of the distance between (i,j) and (i,expected(j))
                expectedJ = lastJ + (i - lastI) * coeff
                vertical_deviation = abs(j - expectedJ)
                new_interval = False
                print(f"{vertical_deviation=}")

                # monotony constraint : if the two previous and the two next anchors are monotonic but not the current
                # the current anchor is discarded
                if num > 1 and num < len(filtered_x) - 2:
                    if (filtered_x[num - 2] <= filtered_x[num - 1] <= filtered_x[num + 1] <= filtered_x[num + 2]) and \
                            (filtered_y[num - 2] <= filtered_y[num - 1] <= filtered_y[num + 1] <= filtered_y[
                                num + 2]) and \
                            (not (filtered_x[num - 1] <= i <= filtered_x[num + 1]) or \
                             not (filtered_y[num - 1] <= j <= filtered_y[num + 1])):
                        print(f"({i},{j}) is ignored (non monotonic)")
                        # the current point is skipped
                        filtered_x[num] = lastI
                        filtered_y[num] = lastJ
                        continue

                # deviated and low density point
                if (vertical_deviation > params[
                    'maxDistToTheDiagonal'] / 2 or i < lastI or j < lastJ) and density_ratio < params[
                    'minDensityRatio']:
                    # localDensity=compute_local_density(i,j,anchor_points,len_sents1,len_sents2,sim_mat,params['deltaX'],params['deltaY'])
                    # deviated point is removed if density is not high enough
                    print(f"({i},{j}) is ignored. Low density : {density_ratio=}")
                    # the current point is skipped
                    filtered_x[num] = lastI
                    filtered_y[num] = lastJ
                    continue

                # only the points that are near the diagonal are taken into account
                if vertical_deviation <= params['maxDistToTheDiagonal']:
                    nb_in_interval += 1
                    lastI = i
                    lastJ = j
                    print(f"({i},{j}) is valid\n")
                else:
                    params['verbose'] and print(
                        f"({i},{j}) is a deviating point {lastI=}, {lastJ=}, {density_ratio=}, {vertical_deviation=}")

                    # considering next points to compute next deviation
                    preview_scope = 2
                    if num + preview_scope < len(filtered_x):
                        (next_i, next_j) = (filtered_x[num + preview_scope], filtered_y[num + preview_scope])
                        next_expectedJ = lastJ + (next_i - lastI) * params['sentRatio']
                        next_vertical_deviation = abs(next_j - next_expectedJ)
                        # the next point is aligned with previous point
                        if next_vertical_deviation <= params['maxDistToTheDiagonal']:
                            params['verbose'] and print(
                                f"({i},{j}) is ignored (next point is aligned with the previous). {vertical_deviation=}")
                            # the current point is skipped
                            filtered_x[num] = lastI
                            filtered_y[num] = lastJ
                            continue
                        else:
                            next_expectedJ = j + (next_i - i) * params['sentRatio']
                            next_vertical_deviation = abs(next_j - next_expectedJ)
                            # if the next point is aligned with the current point, then a new interval should be created
                            if next_vertical_deviation <= params['maxDistToTheDiagonal'] and density_ratio > params[
                                'minDensityRatio']:
                                params['verbose'] and print(
                                    f"({i},{j}) is kept for a new interval because aligned with next points")
                                new_interval = True
                            else:
                                params['verbose'] and print(
                                    f"({i},{j}) is ignored (next point is not aligned) {next_vertical_deviation=} {density_ratio=}")
                                # the current point is skipped
                                filtered_x[num] = lastI
                                filtered_y[num] = lastJ
                                continue
                    # if the deviating point has a high density then create a new interval
                    # ~ # a new interval must be created from the deviating point
                    # ~ if density_ratio > 1.5:
                    # ~ params['verbose'] and print(f"({i},{j}) is kept for a new interval because of high density",density_ratio)
                    # ~ new_interval=True
                    # ~ else:
                    # ~ params['verbose'] and print(f"({i},{j}) is ignored. {density_ratio=}")
                    # ~ # the current point is skipped
                    # ~ filtered_x[num]=lastI
                    # ~ filtered_y[num]=lastJ
                    # ~ continue

                # ~ # computing distance
                d = math.sqrt((i - lastI) ** 2 + (j - lastJ) ** 2)
                # if a there is a gap the previous interval is closed and a new interval will begin
                if d > params['maxGapSize'] and density_ratio > 1.5:
                    params['verbose'] and print(f"{d} > maxGapSize, {density_ratio=}")
                    new_interval = True

                # Creating a new interval if necessary
                if new_interval:
                    endInt = (lastI, lastJ)
                    params['verbose'] and print(d, f"Closing interval ({beginInt},{endInt}) for point ({i},{j})")
                    if beginInt[0] < lastI and beginInt[1] < lastJ:
                        # to save the interval, we compute the density of selected points according to the horizontal width
                        if nb_in_interval / (lastI - beginInt[0]) >= params[
                            'minHorizontalDensity'] and nb_in_interval > 1:
                            intervals.append((beginInt, endInt))
                            interval_length_sent1 += lastI - beginInt[0] + 1
                            interval_length_sent2 += lastJ - beginInt[1] + 1
                            for n in range(max(0, beginInt[0]), lastI + 1):
                                interval_length_char1 += len(sents1[n])
                            for n in range(max(0, beginInt[1]), lastJ + 1):
                                interval_length_char2 += len(sents2[n])
                        else:
                            if params['verbose']:
                                print("Interval", beginInt, endInt, "has been discarded (density too low)")
                    beginInt = (i, j)
                    nb_in_interval = 0

                    if show_plot_4_new_interval:
                        min_x = max(0, i - 100)
                        max_x = min(len(sents1) - 1, i + 100)
                        min_y = max(0, j - 100)
                        max_y = min(len(sents2) - 1, j + 100)

                        x = [point[0] for point in anchor_points if
                             min_x <= point[0] <= max_x and min_y <= point[1] <= max_y]
                        y = [point[1] for point in anchor_points if
                             min_x <= point[0] <= max_x and min_y <= point[1] <= max_y]
                        plt.axis([min_x, max_x, min_y, max_y])
                        plt.title(str(i) + "," + str(j) + '=> new interval')
                        plt.scatter(x, y, c="black", s=1)
                        (i1, j1) = (i - params['deltaX'] / 2, j - params['deltaX'] / 2 - params['deltaY'] / 2)
                        (i1, j2) = (i - params['deltaX'] / 2, j - params['deltaX'] / 2 + params['deltaY'] / 2)
                        (i2, j3) = (i + params['deltaX'] / 2, j + params['deltaX'] / 2 + params['deltaY'] / 2)
                        (i2, j4) = (i + params['deltaX'] / 2, j + params['deltaX'] / 2 - params['deltaY'] / 2)
                        X = [i1, i1, i2, i2, i1]
                        Y = [j1, j2, j3, j4, j1]
                        plt.plot(X, Y, c="grey")
                        plt.show()

                    # la mise à jour de lastI et lastJ ne se fait pas pour
                    # un point déviant n'ayant pas ouvert un intervalle
                    lastI = i
                    lastJ = j
        else:
            lastI = len_sents1 - 1
            lastJ = len_sents2 - 1

        t8 = time.time()
        if params['verbose']:
            print("\n8. Extracting alignable intervals-->", t8 - t7, "s.\n"),

    if lastI != beginInt[0]:
        # closing last interval
        interval_length_sent1 += lastI - beginInt[0] + 1
        interval_length_sent2 += lastJ - beginInt[1] + 1
        for n in range(max(0, beginInt[0]), lastI + 1):
            interval_length_char1 += len(sents1[n])
        for n in range(max(0, beginInt[1]), lastJ + 1):
            interval_length_char2 += len(sents2[n])
        intervals.append((beginInt, (lastI, lastJ)))
        params['verbose'] and print( f"Closing last interval ({beginInt},({lastI},{lastJ}))")

    if params['verbose']:
        print("Total interval length=", interval_length_sent1, "+", interval_length_sent2)
        
    # last filtering step : for each interval, points that are two far from the diagonal are discarded  
    i=0
    for (begin,end) in intervals:
        (x_begin,y_begin)=begin
        (x_end,y_end)=end
        # looking for anchor points in interval begin, end
        while i<len(filtered_x) and filtered_x[i] < x_begin:
            i+=1
        # if point i falls in x interval
        while i<len(filtered_x) and filtered_x[i]>=x_begin and filtered_x[i]<=x_end:
            delete=False
            #  if point i does not fall in y interval, delete point
            if filtered_y[i]<y_begin or filtered_y[i]>y_end:
                delete=True
            expected_y=y_begin+(filtered_x[i]-x_begin)/(x_end-x_begin)*(y_end-y_begin) 
            # if point i is two far from diag, delete point
            if abs((filtered_y[i]-expected_y)/(y_end-y_begin)) > params['localDiagBeam'] or abs(filtered_y[i]-expected_y) > params['maxDistToTheDiagonal']:
                delete=True
                
            if delete :
                if params['veryVerbose']:
                    print(f"point {i} ({filtered_x[i]},{filtered_y[i]}) too far from diagonal")
                del(filtered_x[i])
                del(filtered_y[i])
                if i>=len(filtered_x):
                    break
            else:
                i+=1
        
    return (filtered_x, filtered_y, intervals, interval_length_sent1, interval_length_sent2, interval_length_char1,
            interval_length_char2)


########################################################################## align function
def align(l1,
          l2,
          input_dir,
          file1,
          file2,
          input_format,
          output_dir,
          output_formats,
          output_file_name="",
          col1=0,
          col2=1,
          print_ids=False,
          file_id1="",
          file_id2="",
          add_anchor=False,
          local_params={}):
    global log, split_sent_regex, params

    for param in local_params:
        params[param] = local_params[param]

    if params['useShelve']:
        embed_shelve = shelve.open("embeds")

    if params['splitSent'] and l1 not in split_sent_regex and not params["splitSentRegex"]:
        params['verbose'] and print(f"Default regex ", split_sent_regex["default"],
                                    f"will be used for sentence segmentation in {l1}")
        split_sent_regex[l1] = split_sent_regex['default']
    if params['splitSent'] and l2 not in split_sent_regex and not params["splitSentRegex"]:
        params['verbose'] and print(f"Default regex ", split_sent_regex["default"],
                                    f"will be used for sentence segmentation in {l2}")
        split_sent_regex[l2] = split_sent_regex['default']

    # processing of an aligned file pair
    if params['verbose']:
        print("Processing", file1, "and", file2)

    (sents1, id_sents1, len_sents1, seg2sents1, nb_chars1,pre_anchors_x) = read_input_file(input_dir, file1, input_format, col1, l1)
    (sents2, id_sents2, len_sents2, seg2sents2, nb_chars2,pre_anchors_y) = read_input_file(input_dir, file2, input_format, col2, l2)

    params['verbose'] and print(f"{len(pre_anchors_x)=}, {len(pre_anchors_y)=}")

    # checking if anchors are coherent
    if len(pre_anchors_x) != len(pre_anchors_y) :
        if match_first_pre_anchors:
            if len(pre_anchors_x) > len(pre_anchors_y):
                pre_anchors_x=pre_anchors_x[:len(pre_anchors_y)]
            else:
                pre_anchors_y=pre_anchors_y[:len(pre_anchors_x)]
            print("*************** Prealignment anchor mismatch ! only first ",len(pre_anchors_x)," anchors are kept !")
        else:
            print("*************** Prealignment anchor mismatch ! anchors will be ignored !")
            pre_anchors_x.clear()
            pre_anchors_y.clear()
        
    if len_sents1 * len_sents2 == 0:
        print(f"File is empty ! No sentence read : {len_sents1=} {len_sents2=}")
        return
    # computing output file names
    if output_file_name == "":
        m = re.search(params['filePattern'], file1)

        if m:
            name1 = m.group(1)
            m = re.search(params['filePattern'], file2)
            name2 = m.group(1)
            if name1 != name2:
                name = name1 + "-" + name2
            else:
                name = name1
            output_file_name = name + "." + l1 + "-" + l2
            output_anchor_filename = name + ".anchor." + l1 + "-" + l2
        else:
            output_file_name = os.path.basename(file1) + "-" + os.path.basename(file2)
            output_anchor_filename = file1 + "-" + file2 + ".anchor"
    else:
        output_anchor_filename = output_file_name + ".anchor"

    ####################################################### extract candidate anchor points here !

    # =====> STEP 1-5 : extracting anchor points from similarity matrix

    if params['useNgrams']:
        (points, x, y, sim_mat) = computePointsFromNgrams(sents1, sents2)  
    else:
        (points, x, y, sim_mat, embeds1, embeds2) = computePointsWithEncoder(preprocessor, encoder, sents1, sents2)

    if params['showSimMat']:
        print(sim_mat)
        plt.imshow(sim_mat, vmin=0.2,  cmap='hot',origin='lower')
        plt.show()

    # adding anchor points and deleting mismatching coordinates
    if len(pre_anchors_x) > 0 :
        for x_anchor,y_anchor in zip(pre_anchors_x,pre_anchors_y):
            params['verbose'] and print("Anchor :",x_anchor,y_anchor)
            insertPoint=len(x)
            for i in range(len(x)):
                if x[i] == x_anchor:
                    # deleting old x_anchor,y point 
                    if y[i]!=y_anchor:
                        params['verbose'] and print(f"Conflict with pre anchor Deleting point {[x[i],y[i]]=}")
                        del(points[(x[i],y[i])])
                    break
            
            for j in range(len(y)):
                if y[j] == y_anchor:
                    # deleting old x,y_anchor
                    if x[j]!=x_anchor: 
                        params['verbose'] and print(f"Conflict with pre anchor Deleting point {[x[j],y[j]]=}")
                        if (x[j],y[j]) in points:
                            del(points[(x[j],y[j])])
                    break
             
            # [x_anchor,y_anchor] point has no conflicts
            points[(x_anchor,y_anchor)]=1
        
        # sorting points according to first coordinate
        points={ point:1 for point in sorted(list(points.keys()),key=lambda point:point[0])}

    #######################################################  extract filtered anchor points here !

    # =====> STEP 6-8 : filtering anchor points and extracting alignable intervals



    (filtered_x, filtered_y, intervals, interval_length_sent1, interval_length_sent2, interval_length_char1,
     interval_length_char2) = extract_anchor_points(pre_anchors_x, pre_anchors_y, points, x, y, sents1, sents2, len_sents1, len_sents2, sim_mat)

    if params['adaptativeMode']:
        params['sentRatio'] = interval_length_sent2 / interval_length_sent1
        params['charRatio'] = interval_length_char2 / interval_length_char1
        print(f"Adapted ratios : {sentRatio=} {charRatio=}")
        (filtered_x, filtered_y, intervals, interval_length_sent1, interval_length_sent2, interval_length_char1,
         interval_length_char2) = extract_anchor_points(pre_anchors_x, pre_anchors_y, points, x, y, sents1, sents2, len_sents1, len_sents2, sim_mat)

    if params['writeIntervals'] and len(intervals) > 0:
        output_interval_filename = output_anchor_filename.replace(".anchor", ".intervals") + ".txt"
        f_int = open(output_interval_filename, mode="w", encoding="utf8")
        for interval in intervals:
            (x1, y1) = interval[0]
            (x2, y2) = interval[1]
            # here sentence num starts from 1
            f_int.write(f"{x1 + 1}-{x2 + 1}\t{y1 + 1}-{y2 + 1}\n")
        f_int.close()

    # anchor point output
    if (len(filtered_x) > 0):
        if params['writeAnchorPoints']:
        
            x_final = []
            y_final = []
            score = 0
            nbScore = 0
            for (x2, y2) in zip(filtered_x, filtered_y):
                if sim_mat[x2, y2] >= params['cosThresholdInOutputAnchors']:
                    x_final.append(x2)
                    y_final.append(y2)
                    score += sim_mat[x2, y2]
                    nbScore += 2
            if nbScore > 0:
                mean_score = score / nbScore
                for output_format in output_formats:
                    write_aligned_points(l1, l2, sents1, id_sents1, sents2, id_sents2, x_final, y_final, output_dir,
                                         output_anchor_filename + "." + output_format, output_format, True, print_ids,
                                         mean_score)
            else:
                print("No anchor points over the cos Threshold")


        # display of the points : eliminated points are red
        if print_plot:

            plt.axis([1, len_sents1, 1, len_sents2])
            plt.autoscale()
            plt.title(output_file_name + '.txt - filtered')
            plt.scatter(x, y, c="red", s=1)
            plt.scatter(filtered_x, filtered_y, c="black", s=1)
            for interval in intervals:
                (i1, j1) = interval[0]
                (i2, j2) = interval[1]
                X = [i1, i1, i2, i2, i1]
                Y = [j1, j2, j2, j1, j1]
                plt.plot(X, Y, c="grey")
            if params['savePlot']:
                plt.savefig(os.path.join(output_dir, output_file_name) + '.png')
            if params['showPlot']:
                plt.show()
            plt.close()

        # writing intervals
        if len(intervals) > 0 and params['writeAlignableArea']:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            write_alignable(sents1, id_sents1, intervals, 0, output_dir, file1 + "." + output_format, output_format)
            write_alignable(sents2, id_sents2, intervals, 1, output_dir, file2 + "." + output_format, output_format)

    # If no interval is alignable
    if interval_length_sent1 == 0 or interval_length_sent2 == 0:
        if print_log:
            log.write(
                f"{output_file_name} not alignable\t{l1}={len(sents1)}\t{l2}={len(sents2)}\tmean_score={0}\tsilence1={1:.3f}\tsilence2={1:.3f}\tcommandLine=" + " ".join(
                    sys.argv) + "\n")
        if params['verbose']:
            print(f"{output_file_name} not alignable")
        return

    # =====> STEP 9 : extracting complete alignment using DTW

    if not params['doNotRunDTW']:
        char_ratio = nb_chars2 / nb_chars1 if params['charRatio'] == 0 else params['charRatio']
        params['verbose'] and print("Chararacter ratio=", char_ratio)

       
        (dtw_path, score) = run_dtw(encoder, sents1, sents2, intervals, filtered_x, filtered_y, pre_anchors_x, sim_mat, embeds1,
                                    embeds2, char_ratio)
        # x_dtw and y_dtw contains a list of list of corresponding coordinates
        # eg. x_dtw=[[0],[1,2],[]]
        # eg. y_dtw=[[0],[1],[2]]

        x_dtw = []
        y_dtw = []
        nb_x = 0
        nb_y = 0

        if params['useShelve']:
            encode_hash = embed_shelve
        else:
            encode_hash = {}

        # Chaining the points

        # adding empty pairs at the end
        (last_x, last_y) = dtw_path[-1]
        x_l = list(range(last_x + 1, len_sents1 - 1))
        y_l = list(range(last_y + 1, len_sents2 - 1))

        if len(x_l) > 0:
            x_dtw.append(x_l)
            y_dtw.append([])
            if params['veryVerbose']:
                print(f"Empty pair=([{x_l}],[])")
        if len(y_l) > 0:
            x_dtw.append([])
            y_dtw.append(y_l)
            if params['veryVerbose']:
                print(f"Empty pair=([],[{y_l}])")
        # constitution des groupes en fonctions des bornes
        for i in range(len(dtw_path) - 1, -1, -1):
            if dtw_path[i] != ():
                (x, y) = dtw_path[i]
                # if the point is not empty (interval border)
                if i >= 1 and dtw_path[i - 1] != ():
                    (prev_x, prev_y) = dtw_path[i - 1]
                    x_l = list(range(prev_x + 1, x + 1))
                    y_l = list(range(prev_y + 1, y + 1))

                    if len(x_l) > 0 or len(y_l) > 0:
                        x_dtw.append(x_l)
                        y_dtw.append(y_l)
                    nb_x += len(x_l)
                    nb_y += len(y_l)
                # if the point is the first of the interval, then use (x,y) as a simple point
                else:
                    # ~ x_dtw.append([x])
                    # ~ y_dtw.append([y])
                    # ~ nb_x+=1
                    # ~ nb_y+=1
                    # creating empty pairs for each gap
                    if i - 2 >= 0 and dtw_path[i - 2] != () and print_gap:
                        (prev_x, prev_y) = dtw_path[i - 2]
                        x_l = list(range(prev_x + 1, x + 1))
                        y_l = list(range(prev_y + 1, y + 1))

                        if len(x_l) > 0:
                            x_dtw.append(x_l)
                            y_dtw.append([])
                            if params['veryVerbose']:
                                print(f"Empty pair=([{x_l}],[])")
                        if len(y_l) > 0:
                            x_dtw.append([])
                            y_dtw.append(y_l)
                            if params['veryVerbose']:
                                print(f"Empty pair=([],[{y_l}])")

        # ~ print(f"first x={x},first y={y}")
        # adding first empty pair
        if print_gap:
            x_l = list(range(0, x))
            y_l = list(range(0, y))
            if len(x_l) > 0:
                x_dtw.append(x_l)
                y_dtw.append([])
                if params['veryVerbose']:
                    print(f"Empty pair=([{x_l}],[])")
            if len(y_l) > 0:
                x_dtw.append([])
                y_dtw.append(y_l)
                if params['veryVerbose']:
                    print(f"Empty pair=([],[{y_l}])")

        x_dtw.reverse()
        y_dtw.reverse()

        # grouping points may occur here
        if params['lateGrouping']:
            (x_dtw, y_dtw) = late_grouping(x_dtw, y_dtw, encoder, sents1, sents2, encode_hash, sim_mat, embeds1,
                                           embeds2, char_ratio)

        # writing output files
        mean_score = len(sents1) + len(sents2) - score
        if params['verbose'] and len(sents1) > 0:
            mean_score = 1 - (score / (len(sents1) + len(sents2)))
            print(f"Average similarity={mean_score:.4f}")
        silence1 = (len(sents1) - nb_x) / len(sents1)
        silence2 = (len(sents2) - nb_y) / len(sents2)

        if print_log:
            log.write(
                f"{output_file_name}\t{l1}={len(sents1)}\t{l2}={len(sents2)}\tmean_score={mean_score}\tignored1={(len(sents1) - nb_x)}\tsilence1={silence1:.3f}\tignored2={(len(sents2) - nb_y)}\tsilence2={silence2:.3f}\tcommandLine=" + " ".join(
                    sys.argv) + "\n")
        print(f"{input_format=} {add_anchor=}")
        for output_format in output_formats:
            if output_format == "xml" and input_format == "xml" and add_anchor:
                if not file_id1:
                    file_id1 = l1
                if not file_id2:
                    file_id2 = l2
                print("Add anchors in XML", output_file_name+".xml")
                add_anchor_in_output(input_dir, file1, file2, file_id1, file_id2, x_dtw, y_dtw, output_dir,
                                     params['direction'])
            else:
                write_aligned_points(l1, l2, sents1, id_sents1, sents2, id_sents2, x_dtw, y_dtw, output_dir,
                                     output_file_name + "." + output_format, output_format, False, print_ids,
                                     mean_score, file1, file2)

        if params['useShelve']:
            embed_shelve.close()

        # =====> STEP 10 : parse aligned sentence, extract chunks and align chunk to get word 2 word alignment
       
        output_formats = params.get("outputFormats")
        if params.get('chunkAlignment', True):
            print("Starting Chunk alignment....")
            chunk_alignment(l1, l2, x_dtw, y_dtw, encoder, sents1, sents2, output_file_name, output_dir, output_formats)

        if params.get('wordAlignment', True):
            print("Starting Word alignment....")
            word_alignment(l1, l2, x_dtw, y_dtw, encoder, sents1, sents2, output_file_name, output_dir, output_formats)
        return mean_score

    if params['useShelve']:
        embed_shelve.close()

    # for group [x_inf,..,x_sup], return the interval [x_inf-1,x_sup] (to use in distance_DTW)


# for group [y_inf,..,y_sup], return the interval [y_inf-1,y_sup] (to use in distance_DTW)
def calc_int(group_x, group_y):
    if len(group_x) == 0:
        x_inf = 0
        x_sup = 0
    else:
        x_inf = group_x[0] - 1
        x_sup = group_x[-1]
    if len(group_y) == 0:
        y_inf = 0
        y_sup = 0
    else:
        y_inf = group_y[0] - 1
        y_sup = group_y[-1]
    return (x_inf, x_sup, y_inf, y_sup)


# apply a greedy algorithme to perform the best grouping (which increase sim between source and target)
def late_grouping(x_dtw, y_dtw, encoder, sents1, sents2, encode_hash, sim_mat, embeds1, embeds2, char_ratio):
    # this btree records the index of each group ordered by their gain
    gains = OOBTree()
    groups = []
    # initialisation of the groups data structure : foreach group, record x,y, and the corresponding dist
    for (group_x, group_y) in zip(x_dtw, y_dtw):
        (inf_x, sup_x, inf_y, sup_y) = calc_int(group_x, group_y)
        dist = distance_dtw(encoder, sents1, sents2, encode_hash, {}, sim_mat, embeds1, embeds2, inf_x, sup_x, inf_y,
                            sup_y, char_ratio, False)
        groups.append({'x': group_x, 'y': group_y, "dist": dist})

    # first iteration : for each group, the gain of similarity is computed whether grouping
    # on the left or on the right (direction indicates which direction has the best gain)
    # all the strictly positive gains are recorded in the gains btree
    for i in range(len(groups)):
        compute_gain(gains, groups, i, encoder, sents1, sents2, encode_hash, sim_mat, embeds1, embeds2, char_ratio)

    if len(gains) > 0:
        best_gain = gains.maxKey()
    else:
        best_gain = 0
    # while best grouping produce a positive gain in similarity
    while best_gain > 0:
        i = gains[best_gain][-1]

        group_x = groups[i]['x']
        group_y = groups[i]['y']

        # group i with next group
        if groups[i]['direction'] == 1:
            next_i = next(groups, i)
            params['verbose'] and print(f"group {i} with next {next_i} :", groups[i]['newX'], groups[i]['newY'])
            if next_i != -1:
                # the next group is first "deleted" : dist is set to -1, and x and y are set to []
                groups[next_i]['dist'] = -1
                groups[next_i]['x'] = []
                groups[next_i]['y'] = []
            else:
                print("Wrong direction")
                break
        elif groups[i]['direction'] == -1:
            # group i with previous group
            prev_i = prev(groups, i)
            params['verbose'] and print(f"group {i} with prev {prev_i} :", groups[i]['newX'], groups[i]['newY'])
            if prev_i != -1:
                # the prev group is first "deleted" : dist is set to -1, and x and y are set to []
                groups[prev_i]['dist'] = -1
                groups[prev_i]['x'] = []
                groups[prev_i]['y'] = []
            else:
                print("Wrong direction")
                break
        else:
            print("No direction", i, groups[i])
            break

        # setting the new group with the recorded merging
        groups[i]['x'] = groups[i]['newX']
        groups[i]['y'] = groups[i]['newY']
        groups[i]['dist'] = groups[i]['newDist']

        # update of the gain, on the left and on the right (after the prev group or the next_group which are "deleted")
        compute_gain(gains, groups, i, encoder, sents1, sents2, encode_hash, sim_mat, embeds1, embeds2, char_ratio)

        # update gain on the left and right side
        prev_i = prev(groups, i)
        if prev_i != -1:
            compute_gain(gains, groups, prev_i, encoder, sents1, sents2, encode_hash, sim_mat, embeds1, embeds2,
                         char_ratio)

        next_i = next(groups, i)
        if next_i != -1:
            compute_gain(gains, groups, next_i, encoder, sents1, sents2, encode_hash, sim_mat, embeds1, embeds2,
                         char_ratio)

        # computing best gain for next iteration
        if len(gains) > 0:
            best_gain = gains.maxKey()
        else:
            best_gain = 0

    # returning final groups
    x_dtw = []
    y_dtw = []
    for i, group in enumerate(groups):
        if groups[i]['dist'] != -1:
            x_dtw.append(groups[i]['x'])
            y_dtw.append(groups[i]['y'])
    return (x_dtw, y_dtw)


# compute the gain when grouping on the left (direction=-1) or on the right (direction=1) side
# and record the corresponding merged groups and distance
def compute_gain(gains, groups, i, encoder, sents1, sents2, encode_hash, sim_mat, embeds1, embeds2, char_ratio):
    group_x = groups[i]['x']
    group_y = groups[i]['y']
    dist = groups[i]['dist']

    # removing i for old_gain if any
    if 'gain' in groups[i]:
        old_gain = groups[i]['gain']
        if old_gain > 0 and i in gains[old_gain]:
            gains[old_gain].remove(i)
            # removing the old gain key if necessary
            if len(gains[old_gain]) == 0:
                del (gains[old_gain])

    # no gain is computed for empty groups
    if len(group_x) == 0 or len(group_y) == 0:
        groups[i]['gain'] = 0
        return

    prev_i = prev(groups, i)
    prev_gain = 0
    if prev_i != -1:
        prev_group_x = groups[prev_i]['x']
        prev_group_y = groups[prev_i]['y']
        no_empty = len(prev_group_x) > 0 and len(prev_group_y) > 0
        new_group_x1 = prev_group_x + group_x
        new_group_y1 = prev_group_y + group_y
        (inf_x, sup_x, inf_y, sup_y) = calc_int(new_group_x1, new_group_y1)
        prev_dist = distance_dtw(encoder, sents1, sents2, encode_hash, {}, sim_mat, embeds1, embeds2, inf_x, sup_x,
                                 inf_y, sup_y, char_ratio, False)
        prev_gain = dist - prev_dist
        if no_empty:
            prev_gain -= params['penalty_n_n']
        else:
            prev_gain += params['penalty_0_n']
        # ~ print(i,"prev",no_empty,prev_gain,new_group_x1,new_group_y1,dist,prev_dist,prev_gain)

    next_i = next(groups, i)
    next_gain = 0
    if next_i != -1:
        next_group_x = groups[next_i]['x']
        next_group_y = groups[next_i]['y']
        no_empty = len(next_group_x) > 0 and len(next_group_y) > 0
        new_group_x2 = group_x + next_group_x
        new_group_y2 = group_y + next_group_y
        (inf_x, sup_x, inf_y, sup_y) = calc_int(new_group_x2, new_group_y2)
        next_dist = distance_dtw(encoder, sents1, sents2, encode_hash, {}, sim_mat, embeds1, embeds2, inf_x, sup_x,
                                 inf_y, sup_y, char_ratio, False)
        next_gain = dist - next_dist
        if no_empty:
            next_gain -= params['penalty_n_n']
        else:
            next_gain += params['penalty_0_n']
        # ~ print(i,"next",no_empty,next_gain,new_group_x2,new_group_y2,dist,next_dist,next_gain)

    if next_gain > prev_gain and next_gain > 0:
        groups[i]['gain'] = next_gain
        groups[i]['direction'] = 1
        groups[i]['newX'] = new_group_x2
        groups[i]['newY'] = new_group_y2
        groups[i]['newDist'] = next_dist
        gain = next_gain
    elif prev_gain > 0:
        groups[i]['gain'] = prev_gain
        groups[i]['direction'] = -1
        groups[i]['newX'] = new_group_x1
        groups[i]['newY'] = new_group_y1
        groups[i]['newDist'] = prev_dist
        gain = prev_gain
    else:
        groups[i]['gain'] = 0
        groups[i]['direction'] = 0
        gain = 0

    # updating gains btree
    if gain > 0:
        if not gains.has_key(gain):
            gains[gain] = []
        if i not in gains[gain]:
            gains[gain].append(i)

    return gain


# search for the previous non deleted group before group i
def prev(groups, i):
    if i == 0:
        return -1
    i -= 1
    while i > 0 and groups[i]["dist"] == -1:
        i -= 1
    if groups[i]["dist"] == -1:
        return -1
    else:
        return i


# search for the next non deleted group after group i
def next(groups, i):
    if i == len(groups) - 1:
        return -1
    i += 1
    while i < len(groups) - 1 and groups[i]["dist"] == -1:
        i += 1
    if groups[i]["dist"] == -1:
        return -1
    else:
        return i


# Run the Dynamic time warping algorithm (Viterbi) by computing all the paths
# from each anchor points (the paths must not deviate from these anchors points
# at a distance lower than dtwBeam)

def run_dtw(encoder, sents1, sents2, intervals, filtered_x, filtered_y, pre_anchors_x, sim_mat, embeds1, embeds2, char_ratio):
    global embed_shelve

    path_hash = {}
    dist_hash = {"-2--1;-2--1": 0}  # for the point (-1,-1), the lower bound

    if params['useShelve']:
        encode_hash = embed_shelve
    else:
        encode_hash = {}

    # initialization for the NULL path
    x_first = intervals[0][0][0]
    y_first = intervals[0][0][1]
    
    path_hash[f"{x_first}-{y_first}"] = [[[-1, -1]], 0]
    params['verbose'] and print(f"First point : {x_first}-{y_first}")
    print(f"Init : dtw from ", intervals[0][0], " to ", intervals[-1][1])

    lastBestPath = [[intervals[0][0][0] - 1, intervals[0][0][1] - 1]]
    lastBestScore = 0

    t8 = time.time()
    # process each alignable intervals
    for interval in intervals:
        (x_begin, y_begin) = interval[0]
        (x_end, y_end) = interval[1]
        print (f"Current interval {interval=}")
        key_xy = f"{x_begin}-{y_begin}"
        coeff_y_per_x = (y_end - y_begin) / (x_end - x_begin)

        # these dict allow to drive the paths near the anchor points that are located IN the interval
        x_2_y = {}
        y_2_x = {}
        for i in range(len(filtered_x)):
            x = filtered_x[i]
            y = filtered_y[i]
            if x < x_begin:
                continue
            if x > x_end:
                break
            if y < y_begin or y > y_end:
                continue
            x_2_y[x] = y
            y_2_x[y] = x

        # Bridging the gap between alignable intervals
        # if there is a gap between the last point in path and the first point in current interval, add an empty point () in the path
        if key_xy not in path_hash:
            (lastI, lastJ) = lastBestPath[-1]
            if params['verbose']:
                print(f"Inserting gap between ({lastI},{lastJ}) and ({x_begin},{y_begin})")
            lastBestPath.append(())  # an empty point indicate a break in the path
            lastBestPath.append((x_begin - 1, y_begin - 1))
            path_hash[key_xy] = [lastBestPath, lastBestScore]

        # now run the DTW search between each anchor point in the interval
        # the path are computed recursively, but in order to minimize the recursive depth, the
        # dtw hash is progressively filled by calling the function point by point
        previous_x = x_begin
        previous_y = y_begin
        for x in range(x_begin, x_end + 1):
            localBeam = params['dtwBeam']

            # if (x,y) is an anchor point, run dtw from x !
            if x in x_2_y:
                y = x_2_y[x]
                if params['verbose']:
                    print(f"Anchor point {x},{y}")

                # if it is a preanchor the point cannot be discarded and the local beam is null
                if x in pre_anchors_x:
                    localBeam=0
                else :
                    # computing deviation and beam
                    # if (x,y) is too far from the interval diagonal, it is discarded
                    deviation = 0
                    if y >= y_begin:
                        deviation = abs((y - y_begin) / (y_end - y_begin) - (x - x_begin) / (x_end - x_begin))
                    else:
                        continue

                    # First condition : 1/ deviation > localDiagBeam
                    if (deviation > params['localDiagBeam'] and deviation * (y_end - y_begin) > params['dtwBeam']):
                        del x_2_y[x]
                        if y in y_2_x:
                            del y_2_x[y]
                        if params['verbose']:
                            print(
                                f"deviation*(y_end-y_begin)= {deviation * (y_end - y_begin)} - Anchor point ({x},{y}) is too far from the interval diagonal - point has been discarded!")
                        continue
                    # Second condition : 2/ the ratio between deltaX and deltaY exceeds 4 (1-4 or 4-1 grouping is the max allowed)
                    if (params['noEmptyPair'] and (
                            min(y - previous_y, x - previous_x) == 0 or max(y - previous_y, x - previous_x) / min(
                            y - previous_y, x - previous_x) > 4)):
                        del x_2_y[x]
                        if y in y_2_x:
                            del y_2_x[y]
                        if params['verbose']:
                            print(
                                f"Deviating anchor point ({x},{y}) is too close from the preceding - point has been discarded!")
                        continue

                    # Processing of the gaps (taking into account non monotony) :
                    # if y < previous_y, the area is enlarged : y will be set equal to previous_y and previous_x is decreased, to correspond 
                    # to the last point with x_2_y[prev_x] < y
                    
                    if y < previous_y:
                        print(f"Monotonic discrepancy : {y=} < {previous_y=}. Recomputing previous_x.")
                        prev_x=previous_x
                        # looking for previous point according to y
                        found=False
                        while prev_x > x_begin:
                            prev_x -= 1
                            if prev_x in x_2_y:
                                prev_y = x_2_y[prev_x]
                                if prev_y < y:
                                    y=previous_y
                                    previous_x=prev_x
                                    previous_y=prev_y
                                    found=True
                                    break
                        if not found:
                             y=previous_y
                             previous_x=x_begin
                             previous_y=y_begin     
                
                if params['veryVerbose']:
                    print(f"Running DTW for the point : ({x},{y}) - elapsed from (1,1) =", time.time() - t8, "s.")

   
                # compute the inferior values to give an interval to cut recursion : points that are before
                # x_inf,y_inf should not be considered
                x_inf = previous_x - localBeam
                y_inf = previous_y - localBeam
                
                print( f"Lancement de DTW entre ({max(x_begin, x_inf)},{max(y_begin, y_inf)}) et ({x},{y})")
                (path, dist) = dtw(encoder, sents1, sents2, encode_hash, path_hash, dist_hash, x_2_y, y_2_x,
                                   sim_mat, embeds1, embeds2, x, y, max(x_begin, x_inf), max(y_begin, y_inf), char_ratio)

                if dist == infinite and params['verbose']:
                    print(f"Infinite distance from : ({x},{y})")
                    # initiating a new interval starting from x,y
                    x_begin = x
                    y_begin = y
                    key_xy = f"{x_begin}-{y_begin}"
                    # here creation of a copy of lastBestPath, and addition of the breakpoint
                    lastBestPath = lastBestPath[:]
                    lastBestPath.append(())  # an empty point indicate a break in the path
                    lastBestPath.append((x_begin - 1, y_begin - 1))
                    path_hash[key_xy] = [lastBestPath, lastBestScore]
                    # ~ sys.exit()
                else:
                    lastBestPath = path
                    lastBestScore = dist

                if params['veryVerbose']:
                    print(f"Distance->{dist}")
                previous_x = x
                previous_y = y

        (lastBestPath, lastBestScore) = path_hash[f"{previous_x}-{previous_y}"]

    # chaining with the end of the text
    last_x = len(sents1) - 1
    last_y = len(sents2) - 1
    if (last_x - x) + (last_y - y) < 200:
        if params['verbose']:
            print(f"Last point ({last_x},{last_y})")
        dtw(encoder, sents1, sents2, encode_hash, path_hash, dist_hash, x_2_y, y_2_x, sim_mat, embeds1, embeds2, last_x,
            last_y, x_end, y_end, char_ratio)
    # if last point has not been discarded
    score = infinite
    if f"{last_x}-{last_y}" in path_hash:
        (best_path, score) = path_hash[f"{last_x}-{last_y}"]
    # the last interval is used instead
    if score == infinite:
        (best_path, score) = path_hash[f"{previous_x}-{previous_y}"]

    t9 = time.time()
    if params['verbose']:
        print(f"\n9. Elapsed time for complete DTW-->", t9 - t8, "s.\n")

    return (best_path, score)


# Compute the bestpath (a list of [I,J] pairs) and the corresponding score (the minimum distance)
# The current point correspond to the interval between (infI,inJ) excluded
def dtw(encoder, sents1, sents2, encode_hash, path_hash, dist_hash, x_2_y, y_2_x, sim_mat, embeds1, embeds2, x_end, y_end,
        x_begin, y_begin, char_ratio):
    for i in range(x_begin,x_end+1):
        for j in range(y_begin,y_end+1):
            # The hash path_hash records the result for already computed path, in order to reduce recursivity
            dtw_key = str(i) + "-" + str(j)

            if dtw_key in path_hash:
                continue

            path_by_group = {}
            dist_by_group = {}
            # on examine chaque groupe
            for group in allowed_groups:
                previous_i=i - group[0]
                previous_j=j - group[1]
                previous_key= str(previous_i) + "-" + str(previous_j)
                
                # en principe, previous_key doit être trouvée
                if previous_key in path_hash:
                    (path_by_group[group], dist_by_group[group])=path_hash[previous_key]
                    # ~ print (f"path_hash[{previous_key}]={path_hash[previous_key]}")
                else:
                    # ~ print (f"{previous_key=} pas trouvée")
                    (path_by_group[group], dist_by_group[group])=([], infinite)
                
                # on incrémente la distance pour le groupe courant
                dist_by_group[group] += distance_dtw(encoder, sents1, sents2, encode_hash, dist_hash, sim_mat, embeds1, embeds2,
                                                     previous_i, i, previous_j, j,
                                                     char_ratio) 

            best_group = None
            min_dist = infinite
            for group in allowed_groups:
                if dist_by_group[group] < min_dist:
                    min_dist = dist_by_group[group]
                    best_group = group
            if best_group != None:
                path = path_by_group[best_group][:]  # warning here, create a copy !
                path.append([i, j])
                path_hash[dtw_key] = [path, min_dist]
            else:
                path_hash[dtw_key] = [[], infinite]
   
    return path_hash[str(x_end) + "-" + str(y_end)]



# computing the distance as 1-cosinus
# for empty aligning, dist is equal to distNull which should be near to 1
# when the similarity is below a given threshold (sim_threshold), the dist is fixed to 1 (in order to force using 1-0 or 0-1 pairing)

def distance_dtw(encoder, sents1, sents2, encode_hash, dist_hash, sim_mat, embeds1, embeds2, inf_i, i, inf_j, j,
                 char_ratio, use_coeff=True):
    # if the distance has already been stored in dist_hash
    key = str(inf_i) + "-" + str(i) + ";" + str(inf_j) + "-" + str(j)
    if key in dist_hash:
        return dist_hash[key]

    # coeff indicates the total number of segments (for both language) involved in the alignment
    coeff = 1
    penalty = params['penalty_n_n']

    # case of relations 1-0 et 0-1
    if inf_i == i or inf_j == j:
        return params['distNull'] * coeff

    if i < 0 or j < 0 or inf_i < -2 or inf_j < -2:
        return infinite

    coeff = 2
    if use_encoder:
        # similarity are computed for sentence group
        # case of relations 1-1
        if inf_i == i - 1 and inf_j == j - 1:
            sim = sim_mat[i, j]
            if use_coeff:
                penalty = 0
        # case of relations n-n
        else:
            # calculate embed_i
            if inf_i == i - 1:
                embed_i = embeds1[i][:]
                len_i = len(sents1[inf_i + 1])
            else:
                sent_i = sents1[inf_i + 1]
                for coord_i in range(inf_i + 2, i + 1):
                    sent_i += " " + sents1[coord_i]
                    if use_coeff:
                        coeff += 1
                len_i = len(sent_i)
                if sent_i not in encode_hash:
                    embed_i = encoder.encode([sent_i])
                    embed_i = embed_i / np.linalg.norm(embed_i)  # normalize
                    encode_hash[sent_i] = embed_i
                else:
                    embed_i = encode_hash[sent_i]
            # calculate embed_j
            if inf_j == j - 1:
                embed_j = embeds2[j][:]
                len_j = len(sents2[inf_j + 1])
            else:
                sent_j = sents2[inf_j + 1]
                for coord_j in range(inf_j + 2, j + 1):
                    sent_j += " " + sents2[coord_j]
                    if use_coeff:
                        coeff += 1
                len_j = len(sent_j)
                if sent_j not in encode_hash:
                    embed_j = encoder.encode([sent_j])
                    embed_j = embed_j / np.linalg.norm(embed_j)  # normalize
                    encode_hash[sent_j] = embed_j
                else:
                    embed_j = encode_hash[sent_j]
            sim = float(np.matmul(embed_i, np.transpose(embed_j)))
    else:
        # similarity are computed with vector addition
        # case of relations 1-1 : no penalty
        if inf_i == i - 1 and inf_j == j - 1:
            penalty = 0

        embed_i = embeds1[inf_i + 1][:]
        embed_j = embeds2[inf_j + 1][:]

        len_i = len(sents1[inf_i + 1])
        for coord_i in range(inf_i + 2, i + 1):
            len_i += len(sents1[coord_i])
            embed_i = np.add(embed_i, embeds1[coord_i])
            if use_coeff:
                coeff += 1

        len_j = len(sents2[inf_j + 1])
        for coord_j in range(inf_j + 2, j + 1):
            len_j += len(sents2[coord_j])
            embed_j = np.add(embed_j, embeds2[coord_j])
            if use_coeff:
                coeff += 1
        try:
            norm_i = np.linalg.norm(embed_i)  # normalize
        except:
            norm_i = 0
            for k in range(len(embed_i)):
                norm_i += embed_i[k] ** 2
            norm_i = math.sqrt(norm_i)
            print(f"plantage de linalg.norm, norme calculée directement {norm_i=}")

        try:
            norm_j = np.linalg.norm(embed_j)  # normalize
        except:
            norm_j = 0
            for k in range(len(embed_j)):
                norm_j += embed_j[k] ** 2
            norm_j = math.sqrt(norm_j)
            print(f"plantage de linalg.norm, norme calculée directement {norm_j=}")

        embed_i = embed_i / norm_i  # normalize
        embed_j = embed_j / norm_j  # normalize
        sim = np.matmul(embed_i, np.transpose(embed_j))

    # compute the similarity with neighbouring sentences and substract it to the global sim
    if not params['noMarginPenalty']:
        nb = 0
        nn = 0
        if inf_j >= 0:
            left_embed_j = embeds2[inf_j][:]
            left_sim_j = np.matmul(embed_i, np.transpose(left_embed_j))
            nb += 1
        else:
            left_sim_j = 0
        if j + 1 < len(embeds2):
            right_embed_j = embeds2[j + 1][:]
            right_sim_j = np.matmul(embed_i, np.transpose(right_embed_j))
            nb += 1
        else:
            right_sim_j = 0
        neighbour_sim_j = 0
        if nb > 0:
            neighbour_sim_j = (left_sim_j + right_sim_j) / nb
            nn += 1

        nb = 0
        if inf_i >= 0:
            left_embed_i = embeds1[inf_i][:]
            left_sim_i = np.matmul(left_embed_i, np.transpose(embed_j))
            nb += 1
        else:
            left_sim_i = 0
        if i + 1 < len(embeds1):
            right_embed_i = embeds1[i + 1][:]
            right_sim_i = np.matmul(right_embed_i, np.transpose(embed_j))
            nb += 1
        else:
            right_sim_i = 0
        neighbour_sim_i = 0
        if nb > 0:
            neighbour_sim_i = (left_sim_i + right_sim_i) / nb
            nn += 1

        average_neighbour_sim = 0
        if nn > 0:
            average_neighbour_sim = (neighbour_sim_i + neighbour_sim_j) / nn
        sim -= coeff_neighbour_sim * average_neighbour_sim

    # for empty sentences
    if len_i * len_j == 0:
        return params['distNull'] * coeff

    dist = 1 - sim
    if use_coeff:
        dist += penalty * coeff

    dist = (1 - coeff_sent_len) * dist + coeff_sent_len * lenPenalty(len_i * char_ratio, len_j)

    dist *= coeff
    dist_hash[key] = dist
    return dist


# cf Bertalign
def lenPenalty(len1, len2):
    min_len = min(len1, len2)
    max_len = max(len1, len2)
    return 1 - np.log2(1 + min_len / max_len)


# ************************************************************************* MAIN
if __name__ == "__main__":
    t0 = time.monotonic()

    # processing a unic pair of files
    if input_file1 and input_file2:
        align(l1, l2, input_dir, input_file1, input_file2, input_format, output_dir, output_formats, output_file_name, add_anchor=add_anchor,
              col1=col1, col2=col2, print_ids=print_ids, file_id1=file_id1, file_id2=file_id2)
    # processing a full directory
    else:
        if params['verbose']:
            print("Processing directory", input_dir)
        # reading a tsv file with pairs fileName1 tab fileName2
        if params['inputFileList']:
            f = open(params['inputFileList'], encoding="utf8")
            files1 = []
            files2 = []
            for line in f:
                # skipping comments
                if line[0] != "#":
                    values = line.strip().split("\t")

                    if len(values) == 2:
                        f1 = values[0]
                        f2 = values[1]
                        files1.append(f1)
                        files2.append(f2)
            f.close()
            if params['verbose']:
                print("Files to process", list(zip(files1, files2)))
            for file1, file2 in zip(files1, files2):
                output_file_name = ""
                if params['filePattern'].match(file1):
                    output_file_name = params['filePattern'].match(file1).group(1)
                    l1 = params['filePattern'].match(file1).group(2)
                    l2 = params['filePattern'].match(file2).group(2)
                align(l1, l2, input_dir, file1, file2, input_format, output_dir, output_formats, output_file_name, add_anchor=add_anchor,
                      col1=col1, col2=col2, print_ids=print_ids)
        else:
            files = [f for f in os.listdir(input_dir) if
                     params['filePattern'].match(f)]  # and re.search(input_format+"$",f,re.I)]
            files1 = [f for f in files if params['filePattern'].match(f).group(2) == l1]
            files2 = [f for f in files if params['filePattern'].match(f).group(2) != l1 and (
                        params['filePattern'].match(f).group(2) == l2 or l2 == "*")]
            if params['verbose']:
                print("Files to process", files1)
            # processing input files
            for file1 in files1:
                m = params['filePattern'].match(file1)
                name = m.group(1)
                for file2 in files2:
                    m = params['filePattern'].match(file2)
                    if m.group(1) == name:
                        l2 = m.group(2)
                        align(l1, l2, input_dir, file1, file2, input_format, output_dir, output_formats,
                              output_file_name="", add_anchor=add_anchor, col1=col1, col2=col2, print_ids=print_ids)
    if params['verbose']:
        print("Terminated in", time.monotonic() - t0, "s.")

    if print_log:
        log.close()
