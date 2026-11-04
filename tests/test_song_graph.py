"""Unit and graph tests with a tiny catalog and a fake recommender.

Run from project root: python -m pytest tests/test_song_graph.py -q
"""
import pandas as pd
import pytest

from scripts.song_graph import build_graph, run_request
from scripts.song_lookup import find_song, validate_catalog


@pytest.fixture
def catalog():
    return pd.DataFrame([
        {"track_id": "n1", "track_name": "Numb", "artists": "Linkin Park", "album_name": "Meteora"},
        {"track_id": "n2", "track_name": "Numb", "artists": "Linkin Park", "album_name": "Meteora Bonus Edition"},
        {"track_id": "n3", "track_name": "Numb", "artists": "Another Artist", "album_name": "Other Album"},
        {"track_id": "a1", "track_name": "Another Song", "artists": "Example Artist", "album_name": "Example Album"},
    ])


class FakeRecommender:
    def recommend(self, track_id, top_n=10):
        if track_id == "bad":
            raise ValueError("Unknown track ID: bad")
        return pd.DataFrame([{
            "track_id": "rec1", "track_name": "Recommendation", "artists": "Example",
            "album_name": "Sample", "similarity_score": 0.9,
        }])


@pytest.fixture
def graph(catalog):
    return build_graph(catalog, FakeRecommender())


@pytest.mark.parametrize("user_input,status", [
    ("Recommend songs like Numb", "ambiguous"),
    ("Recommend songs like Numb by Linkin Park", "ambiguous"),
    ("Recommend songs like Numb by Linkin Park from Meteora", "recommended"),
    ("Recommend songs like Numb by Linkin Park from Unknown Album", "not_found"),
    ("Recommend songs like ThisSongDoesNotExist123", "not_found"),
    ("Recommend songs like Numb by ", "invalid_input"),
    ("Recommend songs like Numb by Linkin Park from ", "invalid_input"),
    ("Hello", "invalid_input"),
    ("", "invalid_input"),
])
def test_workflow(graph, user_input, status):
    result = run_request(graph, user_input)

    assert result["status"] == status
    assert len(result["recommendations"]) == (
        1 if status == "recommended" else 0
    )
def test_lookup_case_insensitive(catalog):
    matches = find_song("nUmB", catalog, "linkin park", "meteora")
    assert matches["track_id"].tolist() == ["n1"]


def test_duplicate_metadata_remains_ambiguous(catalog):
    extra = pd.DataFrame([{"track_id": "n4", "track_name": "Numb", "artists": "Linkin Park", "album_name": "Meteora"}])
    graph = build_graph(pd.concat([catalog, extra], ignore_index=True), FakeRecommender())
    assert run_request(graph, "Recommend songs like Numb by Linkin Park from Meteora")["status"] == "ambiguous"


def test_duplicate_track_ids_rejected(catalog):
    invalid = catalog.copy()
    invalid.loc[1, "track_id"] = "n1"
    with pytest.raises(ValueError, match="unique"):
        validate_catalog(invalid)


def test_recommender_value_error(catalog):
    bad_catalog = catalog.copy()
    bad_catalog.loc[0, "track_id"] = "bad"
    graph = build_graph(bad_catalog, FakeRecommender())
    result = run_request(graph, "Recommend songs like Numb by Linkin Park from Meteora")
    assert result["status"] == "recommendation_error"
    assert result["recommendations"] == []
