import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def load_data(path: str) -> pd.DataFraÒme:
    return pd.read_csv(path)


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    return pr_auc, roc_auc


def main():
    data_path = "data/raw/creditcard.csv"
    df = load_data(data_path)

    y = df["Class"]
    X = df.drop(columns=["Class"])

    print("Shape:", df.shape)
    print("Fraud ratio:", y.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("creditcard-fraud-final")

    # ================= LOGISTIC REGRESSION =================
    logreg_model = Pipeline(
        steps=[("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))]
    )

    pr_logreg, roc_logreg = evaluate_model(
        logreg_model, X_train, X_test, y_train, y_test
    )

    # ================= RANDOM FOREST =================
    rf_model = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
    )

    pr_rf, roc_rf = evaluate_model(rf_model, X_train, X_test, y_train, y_test)

    print("LOGREG PR-AUC:", pr_logreg)
    print("RF PR-AUC:", pr_rf)

    # ================= CHOIX DU MEILLEUR =================
    if pr_logreg > pr_rf:
        best_model = logreg_model
        best_name = "LogisticRegression"
        best_pr = pr_logreg
        best_roc = roc_logreg
    else:
        best_model = rf_model
        best_name = "RandomForest"
        best_pr = pr_rf
        best_roc = roc_rf

    print("Best model:", best_name)

    # ================= MLflow Logging =================
    with mlflow.start_run(run_name="best_model_run"):

        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("best_pr_auc", best_pr)
        mlflow.log_metric("best_roc_auc", best_roc)

        # Log du modèle
        mlflow.sklearn.log_model(best_model, artifact_path="best_model")

        # ================= MODEL REGISTRY =================
        model_name = "creditcard_fraud_best_model"
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/best_model"

        mv = mlflow.register_model(model_uri=model_uri, name=model_name)

        print("Registered model:", model_name, "version:", mv.version)

        # ================= PROMOTION AUTOMATIQUE =================
        client = MlflowClient()

        if best_pr > 0.80:
            client.transition_model_version_stage(
                name=model_name, version=mv.version, stage="Production"
            )
            print("Model promoted to Production!")
        else:
            print("Model not promoted (PR-AUC too low).")

        print("BEST PR-AUC:", best_pr)
        print("BEST ROC-AUC:", best_roc)


if __name__ == "__main__":
    main()
