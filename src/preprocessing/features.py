import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering minimal :
    - création de Amount_log1p = log(1 + Amount) pour réduire l'impact des extrêmes
    """
    df = df.copy()

    if "Amount" in df.columns:
        df["Amount_log1p"] = np.log1p(df["Amount"].clip(lower=0))

    return df
