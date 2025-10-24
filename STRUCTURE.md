\# Structure Complète du Projet Scoring Crédit



\## Vue d'ensemble

Projet MLOps complet avec 8 milestones validés selon le cahier des charges OpenClassrooms.



\## Architecture des dossiers



\### `/api/`

api/

├── main.py                 # API FastAPI principale (LightGBM champion)

├── api\_demo.py            # Version démo

├── api\_complete.py        # Version complète

└── test\_api.py           # Tests API



\### `/models/`

models/

├── xgboost\_final\_model.pkl          # Modèle LightGBM champion

├── optimal\_threshold.pkl            # Seuil 9%

├── feature\_columns.pkl              # 354 features

├── model\_metadata.pkl              # Métadonnées

├── shap\_\*.png                      # 12 graphiques SHAP

└── categorical\_encoders.pkl        # Encodeurs



\### `/notebooks/`

reports/

├── data\_drift\_analysis.html        # Rapport Evidently

├── data\_drift\_detailed\_analysis.txt # Analyse complète

├── data\_drift\_by\_feature.png       # Graphiques drift

└── data\_drift\_heatmap.png



\### `/tests/`

tests/

├── test\_simple.py                  # 3 tests unitaires (100% réussite)

├── run\_tests.py                    # Lanceur de tests

└── init.py



\### `/data/`

data/

├── application\_train.csv           # Dataset principal

├── application\_test.csv

├── bureau.csv                      # Tables auxiliaires

└── \[8 autres fichiers CSV]



\## Fichiers racine

\- `README.md` - Documentation principale

\- `requirements.txt` - Dépendances Python (22 packages)

\- `streamlit\_app.py` - Interface utilisateur banquier

\- `main.py` - Backup API



\## Métriques du projet



\### Modèle champion

\- \*\*Algorithme\*\* : LightGBM

\- \*\*AUC\*\* : 78.00%

\- \*\*Seuil optimal\*\* : 9%

\- \*\*Coût métier\*\* : 30,666

\- \*\*Features\*\* : 354



\### MLflow tracking

\- \*\*Expérimentations\*\* : 47 runs

\- \*\*Metrics\*\* : AUC, coût métier, temps

\- \*\*Artifacts\*\* : Modèles, graphiques, rapports



\### Data drift

\- \*\*Features avec drift\*\* : 1/10 (10%)

\- \*\*Niveau\*\* : Drift mineur

\- \*\*Risque\*\* : Faible



\### Tests

\- \*\*Tests unitaires\*\* : 3/3 passent

\- \*\*Couverture\*\* : Seuil, coût métier, performances



\## Status final

✅ PROJET COMPLET - Prêt pour soutenance



\## Commandes utiles

```bash

\# Lancer API

cd api \&\& uvicorn main:app --reload



\# Interface Streamlit  

streamlit run streamlit\_app.py



\# Tests

cd tests \&\& python test\_simple.py



\# MLflow UI

mlflow ui

