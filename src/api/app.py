from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd

# connexion au serveur MLflow
mlflow.set_tracking_uri("http://localhost:5001")

MODEL_NAME = "creditcard_fraud_best_model"

# charger le modèle Production
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/Production")

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API de prédiction de fraude bancaire",
    version="1.0",
)


class Transaction(BaseModel):
    features: list


@app.get("/")
def root():
    return {"message": "Fraud Detection API running"}


@app.post("/predict")
def predict(transaction: Transaction):

    df = pd.DataFrame([transaction.features])

    prediction = model.predict(df)

    return {
        "prediction": int(prediction[0]),
        "label": "fraud" if prediction[0] == 1 else "normal",
    }
