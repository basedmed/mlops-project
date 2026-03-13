import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
)
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing.clean import clean_data
from src.preprocessing.features import build_features


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    return pr_auc, roc_auc


def log_evaluation_artifacts(model, X_test, y_test, artifact_dir="artifacts"):
    os.makedirs(artifact_dir, exist_ok=True)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    pr_path = os.path.join(artifact_dir, "precision_recall_curve.png")
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    roc_path = os.path.join(artifact_dir, "roc_curve.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(pr_path)
    mlflow.log_artifact(roc_path)


def main():
    config = load_config("config.yaml")

    data_path = "data/raw/creditcard.csv"
    df = load_data(data_path)
    df = clean_data(df)
    df = build_features(df)

    y = df["Class"]
    X = df.drop(columns=["Class"])

    print("Shape:", df.shape)
    print("Fraud ratio:", y.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )

    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment(config["experiment_name"])

    logreg_cfg = config["models"]["logistic_regression"]
    rf_cfg = config["models"]["random_forest"]

    logreg_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=logreg_cfg["max_iter"])),
        ]
    )

    pr_logreg, roc_logreg = evaluate_model(
        logreg_model, X_train, X_test, y_train, y_test
    )

    rf_model = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        random_state=config["random_state"],
        n_jobs=-1,
        class_weight=rf_cfg["class_weight"],
    )

    pr_rf, roc_rf = evaluate_model(rf_model, X_train, X_test, y_train, y_test)

    print("LOGREG PR-AUC:", pr_logreg)
    print("RF PR-AUC:", pr_rf)

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

    with mlflow.start_run(run_name="best_model_run"):
        mlflow.log_param("best_model", best_name)
        mlflow.log_param("test_size", config["test_size"])
        mlflow.log_param("random_state", config["random_state"])
        mlflow.log_param("logreg_max_iter", logreg_cfg["max_iter"])
        mlflow.log_param("rf_n_estimators", rf_cfg["n_estimators"])
        mlflow.log_param("rf_class_weight", rf_cfg["class_weight"])

        mlflow.log_metric("best_pr_auc", best_pr)
        mlflow.log_metric("best_roc_auc", best_roc)

        mlflow.sklearn.log_model(best_model, artifact_path="best_model")
        log_evaluation_artifacts(best_model, X_test, y_test)

        model_name = config["promotion"]["registered_model_name"]
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/best_model"

        mv = mlflow.register_model(model_uri=model_uri, name=model_name)

        print("Registered model:", model_name, "version:", mv.version)

        client = MlflowClient()

        if best_pr > config["promotion"]["min_pr_auc"]:
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Production",
            )
            print("Model promoted to Production!")
        else:
            print("Model not promoted (PR-AUC too low).")

        print("BEST PR-AUC:", best_pr)
        print("BEST ROC-AUC:", best_roc)


if __name__ == "__main__":
    main()
