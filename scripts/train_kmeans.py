

"""Evaluate K-Means cluster counts for song audio features."""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "songs_cleaned.csv"

FEATURES = [
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

CLUSTER_COUNTS = [4, 6, 8, 10, 12, 15, 20]
RANDOM_STATE = 42


def evaluate_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Compare candidate cluster counts using fixed preprocessing."""

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])

    results = []

    for k in CLUSTER_COUNTS:
        start = time.perf_counter()

        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            batch_size=1024,
            n_init=10,
        )

        labels = model.fit_predict(X)

        # Sampled silhouette avoids full pairwise-distance costs.
        silhouette = silhouette_score(
            X,
            labels,
            sample_size=min(3000, len(X)),
            random_state=RANDOM_STATE,
        )

        cluster_sizes = np.bincount(labels, minlength=k)

        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette,
            "smallest_cluster": int(cluster_sizes.min()),
            "largest_cluster": int(cluster_sizes.max()),
            "runtime_seconds": round(
                time.perf_counter() - start, 2
            ),
        })

        print(
            f"k={k:2d} | "
            f"silhouette={silhouette:.4f} | "
            f"inertia={model.inertia_:.2f}"
        )

    return pd.DataFrame(results)


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    results = evaluate_clusters(df)

    print("\n=== MODEL COMPARISON ===")
    print(results.to_string(index=False))

    output_dir = ROOT / "reports" / "model_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "kmeans_comparison.csv"
    results.to_csv(output_path, index=False)

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
