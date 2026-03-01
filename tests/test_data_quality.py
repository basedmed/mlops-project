import pandas as pd
import pytest

from src.data.validate_data import validate_dataset


def test_validate_dataset_ok():
    df = pd.DataFrame(
        {
            "Time": [1, 2, 3],
            "Amount": [10.0, 20.0, 30.0],
            "V1": [0.1, 0.2, 0.3],
            "Class": [0, 1, 0],
        }
    )
    validate_dataset(df)


def test_validate_dataset_missing_critical_too_high():
    df = pd.DataFrame(
        {
            "Time": [1, None, None, None, None, None],
            "Amount": [10, 20, 30, 40, 50, 60],
            "Class": [0, 0, 0, 0, 0, 1],
        }
    )
    with pytest.raises(ValueError):
        validate_dataset(df)


def test_validate_dataset_negative_amount():
    df = pd.DataFrame(
        {
            "Time": [1, 2],
            "Amount": [10, -5],
            "Class": [0, 1],
        }
    )
    with pytest.raises(ValueError):
        validate_dataset(df)
