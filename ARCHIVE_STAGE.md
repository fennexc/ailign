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

- `poeme*` : Chenilles et papillons de Gérard Macé ;
- `prose*` : Proche Afrique de Gérard Macé ;
- `Leopardi*` : L’Infinito de Giacomo Leopardi.


## Évaluation lexicale du stage

Les scripts ayant servi à comparer les fichiers TEI de `ref/` et `sys/`, ainsi qu'à explorer les poids de position et de syntaxe, se trouvent encore dans `research-workspace`. Ils ne sont pas recopiés ici pour l'instant. Leur sélection, leur nettoyage minimal et leurs commandes seront ajoutés après vérification.

## Provenance des scripts du stage

Une partie des scripts et modifications réalisés pendant le stage a été écrite avec l'aide de modèles de langage. Le code a ensuite été relu, adapté et testé sur les corpus du stage.
