import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing.clean import clean_data
from src.preprocessing.features import build_features


def test_training_pipeline_runs():
    # Il faut au moins 2 exemples par classe pour stratify
    df = pd.DataFrame(
        {
            "Time": [1, 2, 3, 4, 5, 6],
            "Amount": [10, 20, 30, 40, 50, 60],
            "V1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "Class": [0, 0, 1, 0, 1, 0],
        }
    )

    df = clean_data(df)
    df = build_features(df)

    y = df["Class"]
    X = df.drop(columns=["Class"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )

    model = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
