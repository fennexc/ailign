# Archive du stage — alignement lexical

Ce fichier indique où se trouvent les données utilisées pendant le stage et donne les commandes utiles pour reprendre les calculs. Les commandes ci-dessous sont à lancer depuis la racine du dépôt.

## Organisation des fichiers

```text
archive_stage/
├── raw/      textes bruts ou rétrotraductions avant conversion en TEI
├── tei/      corpus TEI tokenisés utilisés en entrée d'AIlign
├── ref/      alignements lexicaux de référence
├── sys/      sorties d'AIlign à régénérer
└── results/  résultats d'évaluation à régénérer
```

Les dossiers `raw`, `tei` et `ref` regroupent trois ensembles reconnaissables par leur nom :

- `poeme*` : corpus multilingue de poésie ;
- `prose*` : corpus multilingue de prose ;
- `Leopardi*` : original italien et traductions françaises.

Les fichiers `*.1.*` et `*.2.*` correspondent aux deux rétrotraductions disponibles. Le français sert de pivot dans les corpus `poeme` et `prose`.

`sys/` et `results/` sont volontairement vides : les anciennes sorties ne sont pas archivées ici. Elles devront être régénérées pour vérifier les commandes et les scripts.

## Générer un alignement lexical

L'utilisation générale d'AIlign est décrite dans `readme.md`. Voici seulement des commandes correspondant aux données du stage.

### Baseline

```bash
python3 ailign.py \
  --inputFile1 archive_stage/tei/prose.fr.xml \
  --inputFile2 archive_stage/tei/prose.1.de.xml \
  --inputFormat xml --xmlGuide s \
  --outputFormats tei \
  --outputDir archive_stage/sys \
  --outputFileName prose.1.de.baseline \
  --l1 fr --l2 de \
  --wordAlignment \
  --wordAlignmentMethod baseline \
  --wordAlignmentSimilarity embedding=1
```

### Intersection

```bash
python3 ailign.py \
  --inputFile1 archive_stage/tei/prose.fr.xml \
  --inputFile2 archive_stage/tei/prose.1.de.xml \
  --inputFormat xml --xmlGuide s \
  --outputFormats tei \
  --outputDir archive_stage/sys \
  --outputFileName prose.1.de.intersection \
  --l1 fr --l2 de \
  --wordAlignment \
  --wordAlignmentMethod intersection \
  --wordAlignmentSimilarity embedding=1
```

### Union

```bash
python3 ailign.py \
  --inputFile1 archive_stage/tei/prose.fr.xml \
  --inputFile2 archive_stage/tei/prose.1.de.xml \
  --inputFormat xml --xmlGuide s \
  --outputFormats tei \
  --outputDir archive_stage/sys \
  --outputFileName prose.1.de.union \
  --l1 fr --l2 de \
  --wordAlignment \
  --wordAlignmentMethod union \
  --wordAlignmentSimilarity embedding=1
```

### Grow-diag avec position et syntaxe

```bash
python3 ailign.py \
  --inputFile1 archive_stage/tei/prose.fr.xml \
  --inputFile2 archive_stage/tei/prose.1.de.xml \
  --inputFormat xml --xmlGuide s \
  --outputFormats tei \
  --outputDir archive_stage/sys \
  --outputFileName prose.1.de.grow-diag \
  --l1 fr --l2 de \
  --wordAlignment \
  --wordAlignmentMethod grow_diag \
  --wordAlignmentSimilarity embedding=1,position=0.1,syntax=0.1
```

Pour une autre langue ou un autre corpus, remplacer les deux fichiers et les valeurs de `--l1` et `--l2`. Les commandes exactes devront être revérifiées lors de la régénération, notamment pour les langues dont les modèles Stanza posaient problème.

## Scripts d'évaluation présents à la racine

### `eval.all.py`

Lance l'évaluation historique d'AIlign sur les corpus `eval/BAF`, `eval/MD.fr-ar` et `eval/text+berg`, puis écrit les scores dans `eval.log`.

```bash
python3 eval.all.py
```

Ce script ne correspond pas à l'évaluation lexicale TEI réalisée pendant le stage.

### `eval.intervals.py`

Évalue les intervalles du corpus Grimm à partir des chemins inscrits directement dans le script :

```bash
python3 eval.intervals.py
```

Il suppose que `eval/Grimm/ailign/KHM.de-fr.intervals.txt` a déjà été produit.

### `eval.txt.py`

Ce script devait comparer deux alignements textuels :

```bash
python3 eval.txt.py \
  --l1 fr --l2 de \
  --refFile1 CHEMIN_REF_FR --refFile2 CHEMIN_REF_DE \
  --sysFile1 CHEMIN_SYS_FR --sysFile2 CHEMIN_SYS_DE \
  --logFile archive_stage/results/evaluate.tsv
```

Dans son état actuel, des chemins locaux codés en dur remplacent toutefois les arguments de la commande. Il faudra corriger ou retirer ce comportement avant de considérer cette commande comme reproductible.

## Évaluation lexicale du stage

Les scripts ayant servi à comparer les fichiers TEI de `ref/` et `sys/`, ainsi qu'à explorer les poids de position et de syntaxe, se trouvent encore dans `research-workspace`. Ils ne sont pas recopiés ici pour l'instant. Leur sélection, leur nettoyage minimal et leurs commandes seront ajoutés après vérification.

## Provenance des scripts du stage

Une partie des scripts et modifications réalisés pendant le stage a été écrite avec l'aide de modèles de langage. Le code a ensuite été relu, adapté et testé sur les corpus du stage. La régénération des sorties dans `sys/` et `results/` doit servir à vérifier ce qui est effectivement reproductible avant toute réutilisation.
