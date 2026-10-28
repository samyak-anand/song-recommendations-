
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Dataset" / "audio_features_dataset_curated.csv"

AUDIO_FEATURES = [
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]


def analyze_duplicates(df: pd.DataFrame) -> None:
    """Inspect repeated IDs and conflicting audio-feature records."""

    print("\n=== DUPLICATE ANALYSIS ===")

    print("Total rows:", len(df))
    print("Unique track IDs:", df["track_id"].nunique())
    print("Exact duplicate rows:", df.duplicated().sum())

    repeated = df[df.duplicated("track_id", keep=False)]
    print("Rows belonging to repeated track IDs:", len(repeated))
    print("Distinct repeated track IDs:", repeated["track_id"].nunique())

    # Count distinct audio-feature profiles for each track ID.
    profiles = df[["track_id", *AUDIO_FEATURES]].drop_duplicates()

    profile_counts = profiles.groupby("track_id").size()
    conflicting_ids = profile_counts[profile_counts > 1]

    print("Track IDs with conflicting audio profiles:", len(conflicting_ids))
    print("Additional distinct profiles:", int((conflicting_ids - 1).sum()))

    print("\nTop repeated track IDs:")
    print(df["track_id"].value_counts().head(10).to_string())


def analyze_feature_quality(df: pd.DataFrame) -> None:
    """Check missing, infinite and suspicious numeric values."""

    print("\n=== FEATURE QUALITY ===")

    features = df[AUDIO_FEATURES]

    print("Missing numeric values:")
    print(features.isna().sum().to_string())

    print("\nInfinite numeric values:")
    print(
        pd.Series(
            np.isinf(features.to_numpy(dtype=float)).sum(axis=0),
            index=AUDIO_FEATURES,
        ).to_string()
    )

    print("\n=== SPECIAL VALUE CHECKS ===")

    checks = {
        "duration_ms <= 0": df["duration_ms"].le(0),
        "tempo <= 0": df["tempo"].le(0),
        "time_signature == 0": df["time_signature"].eq(0),
        "danceability outside [0,1]": ~df["danceability"].between(0, 1),
        "energy outside [0,1]": ~df["energy"].between(0, 1),
        "valence outside [0,1]": ~df["valence"].between(0, 1),
        "key outside [0,11]": ~df["key"].between(0, 11),
        "mode outside [0,1]": ~df["mode"].between(0, 1),
    }

    for name, condition in checks.items():
        print(f"{name}: {int(condition.sum())}")


def analyze_correlations(df: pd.DataFrame) -> None:
    """Find the strongest absolute Pearson correlations."""

    print("\n=== FEATURE CORRELATIONS ===")

    features = df[AUDIO_FEATURES]
    correlations = features.corr()

    upper_triangle = correlations.where(
        np.triu(np.ones(correlations.shape), k=1).astype(bool)
    )

    strongest = (
        upper_triangle.stack()
        .rename("correlation")
        .reset_index()
    )

    strongest.columns = ["feature_1", "feature_2", "correlation"]
    strongest["absolute_correlation"] = strongest["correlation"].abs()

    strongest = strongest.sort_values(
        "absolute_correlation", ascending=False
    )

    print(strongest.head(15).to_string(index=False))


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH, low_memory=False)

    analyze_duplicates(df)
    analyze_feature_quality(df)
    analyze_correlations(df)

    print("\n=== DATA QUALITY ANALYSIS COMPLETED ===")


if __name__ == "__main__":
    main()
