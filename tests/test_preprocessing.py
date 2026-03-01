import pandas as pd

from src.preprocessing.clean import clean_data
from src.preprocessing.features import build_features


def test_clean_data_removes_duplicates():
    df = pd.DataFrame({"Amount": [1, 1], "Time": [10, 10], "Class": [0, 0]})
    cleaned = clean_data(df)
    assert len(cleaned) == 1


def test_build_features_adds_amount_log1p():
    df = pd.DataFrame({"Amount": [0, 9], "Time": [1, 2], "Class": [0, 1]})
    feats = build_features(df)
    assert "Amount_log1p" in feats.columns
