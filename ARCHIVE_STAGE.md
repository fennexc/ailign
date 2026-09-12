# Archive du stage — alignement lexical

Ce fichier indique où se trouvent les données utilisées pendant le stage et donne les commandes utiles pour reprendre les calculs. Les commandes ci-dessous sont à lancer depuis la racine du dépôt.

## Organisation des fichiers

```text
archive_stage/
├── raw/      textes bruts ou rétrotraductions avant conversion en TEI
├── tei/      corpus TEI tokenisés utilisés en entrée d'AIlign
├── ref/      alignements lexicaux de référence
├── sys/      sorties d'AIlign
└── results/  résultats d'évaluation
```

Les dossiers `raw`, `tei` et `ref` regroupent trois ensembles reconnaissables par leur nom :

- `poeme*` : *Chenilles et papillons* de Gérard Macé ;
- `prose*` : *Proche Afrique* de Gérard Macé ;
- `Leopardi*` : *L’Infinito* de Giacomo Leopardi et ses traductions françaises.

## Produire les alignements

```bash
mkdir -p archive_stage/sys/{prose,poeme,leopardi}

python3 ailign.py --inputFileList archive_stage/tei/filelist_prose.tsv --inputFormat xml --xmlGuide s --outputFormats tei --outputDir archive_stage/sys/prose --l1 fr --l2 fr --wordAlignment --wordAlignmentStrategy grow_diag --wordAlignmentSimilarity embedding=1,position=0.1,syntax=0.1

python3 ailign.py --inputFileList archive_stage/tei/filelist_poeme.tsv --inputFormat xml --xmlGuide s --outputFormats tei --outputDir archive_stage/sys/poeme --l1 fr --l2 fr --wordAlignment --wordAlignmentStrategy grow_diag --wordAlignmentSimilarity embedding=1,position=0.1,syntax=0.1

python3 ailign.py --inputFileList archive_stage/tei/filelist_leopardi.tsv --filePattern '([^/]*)_(it|[^/.]+)[.]xml$' --inputFormat xml --xmlGuide s --outputFormats tei --outputDir archive_stage/sys/leopardi --l1 it --l2 fr --wordAlignment --wordAlignmentStrategy grow_diag --wordAlignmentSimilarity embedding=1,position=0.1,syntax=0.1
```

## Évaluer les alignements

`evaluate_sys.py` compare les liens `corresp` d'une sortie système à ceux de la référence. Il calcule précision, rappel et F1 pour tous les tokens, puis pour les mots pleins.

### Un fichier

```bash
python3 evaluate_sys.py \
  --ref-file archive_stage/ref/prose.1.de.xml \
  --sys-file archive_stage/sys/prose/prose.1.de_word_ai.xml \
  --pivot-file archive_stage/ref/prose.fr.xml \
  --output-prefix archive_stage/results/prose.1.de.grow_diag \
  --details
```

Le script écrit notamment :

```text
archive_stage/results/prose.1.de.grow_diag_evaluation.tsv
archive_stage/results/prose.1.de.grow_diag_by_file.tsv
archive_stage/results/prose.1.de.grow_diag_by_pos.tsv
```

### Tous les fichiers présents dans un dossier système

```bash
python3 evaluate_sys.py \
  --ref-dir archive_stage/ref \
  --sys-dir archive_stage/sys/prose \
  --pivot-file archive_stage/ref/prose.fr.xml \
  --output-prefix archive_stage/results/prose.grow_diag \
  --details
```

La même commande fonctionne pour le poème ou Leopardi en remplaçant le dossier système et le pivot :

- poème : `archive_stage/ref/poeme.fr.xml` ;
- Leopardi : `archive_stage/ref/Leopardi_it.xml`.

## Assistance par modèles de langage

Une partie des scripts et modifications réalisés pendant le stage a été écrite avec l'aide de modèles de langage. Le code a ensuite été relu, adapté et testé sur les corpus du stage.
