"""Exact catalog matching. Track IDs remain authoritative identifiers."""
import pandas as pd

REQUIRED_COLUMNS = ("track_id", "track_name", "artists", "album_name")


def validate_catalog(catalog: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(catalog.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    ids = catalog["track_id"]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("Track IDs must be nonempty and unique")


def find_song(song_name: str, catalog: pd.DataFrame,
              artist_name: str = "", album_name: str = "") -> pd.DataFrame:
    """Find exact title, optional semicolon-separated artist and exact album."""
    columns = list(REQUIRED_COLUMNS)
    titles = catalog["track_name"].fillna("").astype(str).str.strip().str.casefold()
    matches = catalog.loc[titles.eq(song_name.strip().casefold()), columns].copy()

    if artist_name.strip():
        target = artist_name.strip().casefold()
        matches = matches.loc[matches["artists"].fillna("").astype(str).apply(
            lambda value: target in (part.strip().casefold() for part in value.split(";"))
        )]
    if album_name.strip():
        albums = matches["album_name"].fillna("").astype(str).str.strip().str.casefold()
        matches = matches.loc[albums.eq(album_name.strip().casefold())]
    return matches
