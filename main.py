"""Run from project root: python main.py"""
from pathlib import Path

import pandas as pd

from scripts.recommender import SongRecommender  # Keep your existing file unchanged
from scripts.song_graph import build_graph, run_request


def main() -> None:
    root = Path(__file__).resolve().parent
    data_path = root / "data" / "processed" / "songs_cleaned.csv"
    catalog = pd.read_csv(data_path)
    recommender = SongRecommender(data_path=data_path, model_dir=root / "models")
    graph = build_graph(catalog, recommender)

    print("Song recommendations. Type 'exit' to quit.")
    print("Example: Recommend songs like Numb by Linkin Park from Meteora")
    while True:
        try:
            request = input("\nSong request: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if request.casefold() in {"exit", "quit"}:
            break
        result = run_request(graph, request)
        print(f"Status: {result['status']}")
        print(result["response"])
        for index, song in enumerate(result["recommendations"], start=1):
            print(f"{index}. {song['track_name']} — {song['artists']} "
                  f"(score: {song['similarity_score']:.3f})")


if __name__ == "__main__":
    main()
