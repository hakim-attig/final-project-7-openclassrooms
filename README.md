\#   Projet Scoring Crédit - Prêt à Dépenser



\## Description

Système d'évaluation automatique de crédit utilisant l'intelligence artificielle pour prédire le risque de défaut des clients.



\## Modèle Champion

\- \*\*Algorithme\*\* : LightGBM

\- \*\*Performance\*\* : AUC 78.00%

\- \*\*Seuil optimal\*\* : 9%

\- \*\*Coût optimal\*\* : 30,666



\## Structure du Projet



projet\_scoring\_credit/

├── notebooks/              # Analyses et développement

├── api/                     # API FastAPI

├── models/                  # Modèles entraînés

├── tests/                   # Tests unitaires

├── reports/                 # Rapports d'analyse

└── README.md               # Ce fichier



\## Installation

```bash

pip install -r requirements.txt



Utilisation



Lancer l'API : cd api \&\& uvicorn main:app --reload

Interface : streamlit run streamlit\_app.py

Tests : cd tests \&\& python test\_simple.py



Résultats



8 milestones validés

Data drift analysé avec Evidently

Tests unitaires 100% réussis

Modèle LightGBM champion déployé



Auteur

Hakim ATTIG - Projet 7 OpenClassrooms: Master 2 Data Scientist 

