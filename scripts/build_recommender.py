
"""Build artifacts for the content-based song recommender."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "songs_cleaned.csv"
MODEL_DIR = PROJECT_ROOT / "models"

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


def load_dataset() -> pd.DataFrame:
    """Load and validate the cleaned song dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("The cleaned dataset is empty.")

    if df["track_id"].isna().any():
        raise ValueError("Missing track IDs detected.")

    if df["track_id"].duplicated().any():
        raise ValueError("Duplicate track IDs detected.")

    if df[AUDIO_FEATURES].isna().any().any():
        raise ValueError("Missing audio features detected.")

    return df


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[np.ndarray, StandardScaler]:
    """Scale audio features and normalize for cosine similarity."""
    X = df[AUDIO_FEATURES].to_numpy(dtype=np.float64)

    if not np.isfinite(X).all():
        raise ValueError("Non-finite audio feature values detected.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_normalized = normalize(X_scaled).astype(np.float32)

    return X_normalized, scaler


def save_artifacts(
    df: pd.DataFrame,
    vectors: np.ndarray,
    scaler: StandardScaler,
) -> None:
    """Persist the feature matrix and its matching metadata."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "features": AUDIO_FEATURES,
        "track_ids": df["track_id"].tolist(),
        "scaler": scaler,
    }

    joblib.dump(
        metadata,
        MODEL_DIR / "recommender_metadata.joblib",
    )

    np.save(
        MODEL_DIR / "song_vectors.npy",
        vectors,
    )


def main() -> None:
    df = load_dataset()
    vectors, scaler = build_feature_matrix(df)
    save_artifacts(df, vectors, scaler)

    print("\n=== STAGE 1.5: RECOMMENDER BUILD ===")
    print(f"Total songs: {len(df):,}")
    print(f"Audio features: {len(AUDIO_FEATURES)}")
    print(f"Feature matrix: {vectors.shape}")
    print(f"NaN values: {np.isnan(vectors).sum()}")
    print(f"Infinite values: {np.isinf(vectors).sum()}")
    print(f"Artifacts saved in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
