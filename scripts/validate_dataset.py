from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "Dataset" / "audio_features_dataset_curated.csv"


def validate_dataset(path: Path) -> pd.DataFrame:
    """Inspect the dataset without modifying the original file."""

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    print("\n=== DATASET OVERVIEW ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\n=== COLUMN NAMES AND TYPES ===")
    print(df.dtypes.to_string())

    print("\n=== SAMPLE RECORDS ===")
    print(df.head(5).to_string(index=False))

    print("\n=== MISSING VALUES ===")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": df.isna().mean().mul(100).round(2),
    })
    print(missing.to_string())

    print("\n=== DUPLICATE ROWS ===")
    print(f"Exact duplicates: {df.duplicated().sum():,}")

    for column in ("track_id", "id"):
        if column in df.columns:
            print(
                f"Duplicate {column}: "
                f"{df[column].duplicated().sum():,}"
            )

    print("\n=== NUMERIC FEATURE SUMMARY ===")
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        print(numeric.describe().T.to_string())
    else:
        print("No numeric columns found.")

    print("\n=== VALIDATION COMPLETED ===")

    return df


if __name__ == "__main__":
    validate_dataset(DATASET_PATH)
