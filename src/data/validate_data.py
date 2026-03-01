import sys
import pandas as pd

CRITICAL_COLS = ["Time", "Amount"]
MAX_MISSING_RATE = 0.05  # 5%


def validate_dataset(df: pd.DataFrame) -> None:
    # Colonnes obligatoires
    required = set(CRITICAL_COLS + ["Class"])
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    # Taux de missing sur colonnes critiques
    miss_rate = df[CRITICAL_COLS].isna().any(axis=1).mean()
    if miss_rate > MAX_MISSING_RATE:
        raise ValueError(
            "Data Quality Gate failed: missing critical rate="
            f"{miss_rate:.3%} > {MAX_MISSING_RATE:.0%}"
        )

    # Amount négatif interdit
    if (df["Amount"] < 0).any():
        raise ValueError("Negative Amount detected")

    # Class doit être 0 ou 1
    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("Invalid values in Class column")


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_data.py <path_to_csv>")
        sys.exit(1)

    path = sys.argv[1]
    df = pd.read_csv(path)
    validate_dataset(df)
    print("✅ Data Quality Gate: OK")


if __name__ == "__main__":
    main()
