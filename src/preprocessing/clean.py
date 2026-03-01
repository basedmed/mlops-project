import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage minimal (pédagogique) :
    - copie du df
    - suppression des doublons
    - remplissage des NaN (ici 0, pour garder un pipeline robuste)
    """
    df = df.copy()
    df = df.drop_duplicates()
    df = df.fillna(0)
    return df
