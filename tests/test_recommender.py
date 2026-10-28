
"""Automated tests for the Stage 1.5 song recommender."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from recommender import SongRecommender


@pytest.fixture(scope="module")
def engine():
    """Load the recommender once for all tests."""
    return SongRecommender()


@pytest.fixture(scope="module")
def query_id(engine):
    return engine.songs.iloc[0]["track_id"]


def test_unique_dataset_track_ids(engine):
    assert engine.songs["track_id"].is_unique


def test_feature_matrix_shape(engine):
    assert engine.vectors.shape == (
        len(engine.songs),
        len(engine.metadata["features"]),
    )


def test_feature_vectors_are_finite(engine):
    assert np.isfinite(engine.vectors).all()


def test_feature_vectors_are_normalized(engine):
    norms = np.linalg.norm(engine.vectors, axis=1)

    # Zero vectors, if present, cannot be normalized
    # to unit length and need separate handling.
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)


def test_recommendation_count(engine, query_id):
    results = engine.recommend(query_id, top_n=10)
    assert len(results) == 10


def test_input_track_excluded(engine, query_id):
    results = engine.recommend(query_id)

    assert query_id not in results["track_id"].values


def test_unique_recommendation_track_ids(engine, query_id):
    results = engine.recommend(query_id)

    assert results["track_id"].is_unique


def test_unique_title_artist_pairs(engine, query_id):
    results = engine.recommend(query_id)

    keys = [
        SongRecommender._song_key(
            row["track_name"],
            row["artists"],
        )
        for _, row in results.iterrows()
    ]

    assert len(keys) == len(set(keys))


def test_query_song_title_artist_excluded(engine, query_id):
    results = engine.recommend(query_id)
    query = engine.songs.iloc[engine.id_to_index[query_id]]

    query_key = SongRecommender._song_key(
        query["track_name"],
        query["artists"],
    )

    result_keys = {
        SongRecommender._song_key(
            row["track_name"],
            row["artists"],
        )
        for _, row in results.iterrows()
    }

    assert query_key not in result_keys


def test_similarity_scores_sorted(engine, query_id):
    results = engine.recommend(query_id)

    scores = results["similarity_score"].to_numpy()

    assert np.all(np.diff(scores) <= 1e-6)


def test_similarity_scores_in_valid_range(engine, query_id):
    results = engine.recommend(query_id)

    scores = results["similarity_score"].to_numpy()

    assert np.all(scores >= -1.00001)
    assert np.all(scores <= 1.00001)


def test_recommendations_are_deterministic(engine, query_id):
    first = engine.recommend(query_id)
    second = engine.recommend(query_id)

    assert first.equals(second)


def test_unknown_track_id(engine):
    with pytest.raises(ValueError, match="Unknown track ID"):
        engine.recommend("nonexistent_track_id")


@pytest.mark.parametrize("top_n", [0, -1, 1.5, True])
def test_invalid_top_n(engine, query_id, top_n):
    with pytest.raises(ValueError):
        engine.recommend(query_id, top_n=top_n)


def test_top_n_one(engine, query_id):
    results = engine.recommend(query_id, top_n=1)

    assert len(results) == 1
