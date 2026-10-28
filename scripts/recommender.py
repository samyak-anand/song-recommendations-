
"""
Stage 1.5 — Content-Based Song Recommendation Engine.

Recommend songs using cosine similarity between standardized
audio-feature vectors.

Features:
- Loads saved preprocessing artifacts
- Validates dataset and vector alignment
- Calculates cosine similarity
- Excludes the input song
- Prevents duplicate title-artist recommendations
- Returns ranked recommendations with similarity scores
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "songs_cleaned.csv"
MODEL_DIR = ROOT / "models"

METADATA_PATH = MODEL_DIR / "recommender_metadata.joblib"
VECTORS_PATH = MODEL_DIR / "song_vectors.npy"

RESULT_COLUMNS = [
    "track_id",
    "track_name",
    "artists",
    "album_name",
]


# ============================================================
# 2. SONG RECOMMENDATION ENGINE
# ============================================================

class SongRecommender:
    """Content-based recommendation engine using cosine similarity."""

    def __init__(
        self,
        data_path: Path = DATA_PATH,
        model_dir: Path = MODEL_DIR,
    ):
        """Load and validate the dataset and saved artifacts."""

        data_path = Path(data_path)
        model_dir = Path(model_dir)

        metadata_path = model_dir / "recommender_metadata.joblib"
        vectors_path = model_dir / "song_vectors.npy"

        # Check required files
        for path in [data_path, metadata_path, vectors_path]:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required file not found: {path}"
                )

        # Load dataset
        self.songs = pd.read_csv(data_path)

        # Load trusted, locally generated artifacts only.
        self.metadata = joblib.load(metadata_path)

        self.vectors = np.load(
            vectors_path,
            allow_pickle=False,
        )

        # Validate metadata and dataset alignment
        saved_ids = self.metadata["track_ids"]
        current_ids = self.songs["track_id"].tolist()

        if current_ids != saved_ids:
            raise ValueError(
                "Dataset does not match saved model artifacts."
            )

        if self.songs["track_id"].isna().any():
            raise ValueError("Missing track IDs detected.")

        if self.songs["track_id"].duplicated().any():
            raise ValueError("Duplicate track IDs detected.")

        expected_shape = (
            len(self.songs),
            len(self.metadata["features"]),
        )

        if self.vectors.shape != expected_shape:
            raise ValueError(
                f"Invalid vector shape: {self.vectors.shape}. "
                f"Expected: {expected_shape}"
            )

        if not np.isfinite(self.vectors).all():
            raise ValueError(
                "Feature vectors contain NaN or infinite values."
            )

        # Map track IDs to their row positions
        self.id_to_index = {
            track_id: index
            for index, track_id in enumerate(saved_ids)
        }

        # Precompute normalized song identifiers for deduplication
        self.song_keys = [
            self._song_key(row["track_name"], row["artists"])
            for _, row in self.songs.iterrows()
        ]

    # ========================================================
    # 3. HELPER FUNCTION
    # ========================================================

    @staticmethod
    def _song_key(track_name, artists) -> tuple[str, str]:
        """
        Create a normalized title-artist identifier.

        Different track IDs with the same title and artist
        are treated as alternative releases of one song.
        """
        return (
            str(track_name).strip().casefold(),
            str(artists).strip().casefold(),
        )

    # ========================================================
    # 4. RECOMMENDATION LOGIC
    # ========================================================

    def recommend(
        self,
        track_id: str,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Return the top-N unique songs ranked by cosine similarity.

        Args:
            track_id: Track ID of the input song.
            top_n: Maximum number of recommendations.

        Returns:
            DataFrame containing song metadata and similarity scores.
        """

        # Validate input track
        if track_id not in self.id_to_index:
            raise ValueError(
                f"Unknown track ID: {track_id}"
            )

        # Validate requested recommendation count
        if type(top_n) is not int or not 1 <= top_n < len(self.songs):
            raise ValueError(
                "top_n must be between 1 and N-1."
            )

        # Identify input song
        query_index = self.id_to_index[track_id]
        query_vector = self.vectors[query_index]

        # Cosine similarity = dot product of normalized vectors
        scores = self.vectors @ query_vector

        # Exclude input track
        scores[query_index] = -np.inf

        # Rank all candidates by descending similarity.
        # Stable sort preserves dataset order for score ties.
        ranked_indices = np.argsort(
            -scores,
            kind="stable",
        )

        # Exclude alternate releases of the input song
        seen_songs = {self.song_keys[query_index]}
        selected_indices = []

        # Select unique recommendations
        for index in ranked_indices:

            index = int(index)

            if index == query_index:
                continue

            song_key = self.song_keys[index]

            if song_key in seen_songs:
                continue

            seen_songs.add(song_key)
            selected_indices.append(index)

            if len(selected_indices) >= top_n:
                break

        # Build result DataFrame
        results = self.songs.iloc[
            selected_indices
        ][RESULT_COLUMNS].copy()

        results["similarity_score"] = scores[
            selected_indices
        ]

        return results.reset_index(drop=True)


# ============================================================
# 5. EXECUTION / SMOKE TEST
# ============================================================

def main() -> None:
    """Demonstrate the recommendation engine."""

    recommender = SongRecommender()

    # Use the first song as a reproducible test query
    example_song = recommender.songs.iloc[0]
    example_id = example_song["track_id"]

    print("\n=== STAGE 1.5: SONG RECOMMENDATION ENGINE ===")

    print("\n=== INPUT SONG ===")
    print("Title:", example_song["track_name"])
    print("Artist:", example_song["artists"])
    print("Track ID:", example_id)

    recommendations = recommender.recommend(
        track_id=example_id,
        top_n=10,
    )

    print("\n=== TOP 10 RECOMMENDATIONS ===")

    print(
        recommendations[
            [
                "track_name",
                "artists",
                "similarity_score",
            ]
        ].to_string(index=False)
    )

    print("\n=== VALIDATION ===")
    print("Recommendations:", len(recommendations))

    print(
        "Unique track IDs:",
        recommendations["track_id"].nunique(),
    )

    unique_songs = {
        SongRecommender._song_key(
            row["track_name"],
            row["artists"],
        )
        for _, row in recommendations.iterrows()
    }

    print("Unique title-artist pairs:", len(unique_songs))

    print(
        "Input track excluded:",
        example_id not in recommendations["track_id"].values,
    )


if __name__ == "__main__":
    main()
