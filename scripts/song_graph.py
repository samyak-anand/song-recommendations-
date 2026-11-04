"""Three-node LangGraph workflow for deterministic song recommendation."""
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from scripts.song_lookup import find_song, validate_catalog


class SongState(TypedDict):
    user_input: str
    song: str
    requested_artist: str
    requested_album: str
    track_id: str
    artist: str
    album: str
    status: str
    response: str
    recommendations: list[dict[str, Any]]


def make_initial_state(user_input: str) -> SongState:
    return {
        "user_input": user_input, "song": "", "requested_artist": "",
        "requested_album": "", "track_id": "", "artist": "",
        "album": "", "status": "", "response": "", "recommendations": [],
    }


def identify_song(state: SongState) -> dict:
    """Parse a song title with optional artist and album."""

    query = state["user_input"].strip()

    prefix = "recommend songs like "
    if query.casefold().startswith(prefix):
        query = query[len(prefix):].strip()

    if not query or query.casefold().endswith((" by", " from")):
        return {
            "status": "invalid_input",
            "response": "Enter a song title, optionally followed by artist and album."
        }

    song = query
    artist = ""
    album = ""

    # Extract album only when it follows an artist.
    by_pos = query.casefold().rfind(" by ")

    if by_pos != -1:
        song = query[:by_pos].strip()
        remaining = query[by_pos + 4:].strip()

        from_pos = remaining.casefold().rfind(" from ")

        if from_pos != -1:
            artist = remaining[:from_pos].strip()
            album = remaining[from_pos + 6:].strip()
        else:
            artist = remaining

    if not song or (by_pos != -1 and not artist) or (
        by_pos != -1 and from_pos != -1 and not album
    ):
        return {
            "status": "invalid_input",
            "response": "Please provide complete song, artist and album information."
        }

    return {
        "song": song,
        "requested_artist": artist,
        "requested_album": album,
        "status": "pending",
        "response": ""
    }
def build_graph(catalog: pd.DataFrame, recommender: Any):
    """Compile one graph, injecting catalog and the existing recommender."""
    validate_catalog(catalog)

    def lookup_song(state: SongState) -> dict:
        matches = find_song(state["song"], catalog,
                            state["requested_artist"], state["requested_album"])
        if matches.empty:
            return {"status": "not_found", "response":
                    f"No matching track found for '{state['song']}'."}
        if len(matches) > 1:
            choices = matches[["track_name", "artists", "album_name"]].drop_duplicates().head(10)
            options = "; ".join(
                f"{row.track_name} — {row.artists} (Album: {row.album_name})"
                for row in choices.itertuples(index=False)
            )
            note = ("Multiple track IDs share the same title, artist and album; "
                    "this request cannot be uniquely resolved."
                    if len(choices) == 1
                    else "Specify the artist or album to narrow the match.")
            return {"status": "ambiguous", "response":
                    f"Found {len(matches)} matching tracks. {note} Options: {options}"}
        track = matches.iloc[0]
        return {"track_id": str(track["track_id"]), "artist": str(track["artists"]),
                "album": str(track["album_name"]), "status": "found",
                "response": f"Found '{track['track_name']}' by {track['artists']}."}

    def recommend_songs(state: SongState) -> dict:
        try:
            results = recommender.recommend(track_id=state["track_id"], top_n=10)
        except ValueError as error:
            return {"recommendations": [], "status": "recommendation_error",
                    "response": str(error)}
        records = results.to_dict(orient="records")
        return {"recommendations": records, "status": "recommended",
                "response": (f"Generated {len(records)} recommendations for "
                             f"'{state['song']}' by {state['artist']} "
                             f"from the album '{state['album']}'.")}

    builder = StateGraph(SongState)
    builder.add_node("identify_song", identify_song)
    builder.add_node("lookup_song", lookup_song)
    builder.add_node("recommend_songs", recommend_songs)
    builder.add_edge(START, "identify_song")
    builder.add_conditional_edges("identify_song",
                                  lambda state: "stop" if state["status"] == "invalid_input" else "lookup",
                                  {"lookup": "lookup_song", "stop": END})
    builder.add_conditional_edges("lookup_song",
                                  lambda state: "recommend" if state["status"] == "found" else "stop",
                                  {"recommend": "recommend_songs", "stop": END})
    builder.add_edge("recommend_songs", END)
    return builder.compile()


def run_request(graph: Any, user_input: str) -> SongState:
    """Run a request and return the complete state (no printing side effects)."""
    return graph.invoke(make_initial_state(user_input))
