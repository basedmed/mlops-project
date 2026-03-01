import pandas as pd
import mlflow


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main():
    # 1) Connexion au tracking MLflow (serveur Docker)
    mlflow.set_tracking_uri("http://localhost:5001")

    # 2) Nom du modèle dans le Model Registry
    model_name = "creditcard_fraud_best_model"

    # 3) Charger la version "Production"
    # (même si stages seront dépréciés plus tard, pour ton projet c'est OK)
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.pyfunc.load_model(model_uri)

    # 4) Charger des données (exemple : on prend 5 lignes)
    df = load_data("data/raw/creditcard.csv")

    # On enlève la colonne cible si elle existe
    if "Class" in df.columns:
        X = df.drop(columns=["Class"])
        y_true = df["Class"]
    else:
        X = df
        y_true = None

    sample = X.head(5)

    # 5) Prédiction (classe 0/1)
    preds = model.predict(sample)

    print("=== Predictions (0=normal, 1=fraud) ===")
    print(preds)

    # 6) Si tu veux aussi afficher les vraies classes (sur ces 5 lignes)
    if y_true is not None:
        print("\n=== True labels ===")
        print(y_true.head(5).values)


if __name__ == "__main__":
    main()
