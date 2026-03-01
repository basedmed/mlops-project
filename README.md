# Projet MLOps – Détection de fraude (Credit Card)

## Objectif
Construire un pipeline MLOps complet pour la détection de fraude sur transactions carte bancaire :
- Entraîner plusieurs modèles
- Suivre les expériences (MLflow)
- Sélectionner automatiquement le meilleur modèle
- Enregistrer le modèle dans le Model Registry (versioning)
- Promouvoir automatiquement en Production si la performance est suffisante
- Charger le modèle Production et effectuer des prédictions

---

## Structure du projet

```text
mlops-project/
├── data/
│   ├── raw/                 # Données brutes (creditcard.csv)
│   └── processed/           # (optionnel) Données preprocessées
├── docker/
│   └── Dockerfile
├── mlflow/                  # Stockage MLflow (DB + artefacts)
│   ├── mlflow.db
│   └── mlruns/
├── src/head -n 20 README.md
│   ├── train.py             # Training + MLflow tracking + registry
│   └── predict.py           # Chargement du modèle Production + prédiction
├── docker-compose.yml       # Serveur MLflow
├── requirements.txt
└── README.md