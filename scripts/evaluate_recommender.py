
"""Evaluate recommendation behavior across sampled songs."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from recommender import SongRecommender


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "model_evaluation"

SAMPLE_SIZE = 100
TOP_N = 10
RANDOM_STATE = 42


def evaluate() -> None:
    engine = SongRecommender()

    sample = engine.songs.sample(
        n=min(SAMPLE_SIZE, len(engine.songs)),
        random_state=RANDOM_STATE,
    )

    records = []

    for _, query in sample.iterrows():
        start = perf_counter()

        recommendations = engine.recommend(
            query["track_id"],
            top_n=TOP_N,
        )

        latency_ms = (perf_counter() - start) * 1000

        query_artist = str(query["artists"]).strip().casefold()

        same_artist_count = sum(
            str(artist).strip().casefold() == query_artist
            for artist in recommendations["artists"]
        )

        records.append({
            "query_track_id": query["track_id"],
            "query_title": query["track_name"],
            "latency_ms": latency_ms,
            "recommendation_count": len(recommendations),
            "mean_similarity": recommendations[
                "similarity_score"
            ].mean(),
            "same_artist_count": same_artist_count,
        })

    results = pd.DataFrame(records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "recommendation_evaluation.csv"
    results.to_csv(output_path, index=False)

    print("\n=== STAGE 1.6: RECOMMENDATION EVALUATION ===")
    print("Queries evaluated:", len(results))
    print("Top-N:", TOP_N)

    print(
        "Mean latency (ms):",
        round(results["latency_ms"].mean(), 2),
    )

    print(
        "P95 latency (ms):",
        round(
            np.percentile(results["latency_ms"], 95),
            2,
        ),
    )

    print(
        "Mean similarity:",
        round(results["mean_similarity"].mean(), 4),
    )

    print(
        "Average same-artist recommendations:",
        round(results["same_artist_count"].mean(), 2),
    )

    print("Report:", output_path)


if __name__ == "__main__":
    evaluate()
    