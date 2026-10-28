
"""Prepare unique, valid song records for clustering."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "Dataset" / "audio_features_dataset_curated.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate IDs and invalid audio records."""

    cleaned = df.copy()

    # Retain one record for each track ID.
    cleaned = cleaned.drop_duplicates(
        subset="track_id",
        keep="first",
    )

    # Exclude records with invalid essential values.
    valid = (
        cleaned["duration_ms"].gt(0)
        & cleaned["tempo"].gt(0)
        & cleaned["time_signature"].gt(0)
    )

    cleaned = cleaned.loc[valid].copy()

    # Remove rows with non-finite model features.
    finite = np.isfinite(
        cleaned[AUDIO_FEATURES].to_numpy(dtype=float)
    ).all(axis=1)

    cleaned = cleaned.loc[finite].reset_index(drop=True)

    return cleaned


def prepare_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, StandardScaler]:
    """Fit a scaler and transform the audio features."""

    scaler = StandardScaler()

    X = scaler.fit_transform(df[AUDIO_FEATURES])

    return X, scaler


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    cleaned = clean_dataset(df)
    X, _ = prepare_features(cleaned)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_path = OUTPUT_DIR / "songs_cleaned.csv"

    cleaned.to_csv(cleaned_path, index=False)

    print("\n=== PREPROCESSING SUMMARY ===")
    print(f"Original records: {len(df):,}")
    print(f"Cleaned records: {len(cleaned):,}")
    print(f"Removed records: {len(df) - len(cleaned):,}")

    print("\n=== FEATURE MATRIX ===")
    print("Shape:", X.shape)
    print("Features:", AUDIO_FEATURES)
    print("NaN values:", int(np.isnan(X).sum()))
    print("Infinite values:", int(np.isinf(X).sum()))

    print("\nCleaned dataset saved to:", cleaned_path)


if __name__ == "__main__":
    main()
