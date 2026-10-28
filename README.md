# Song Recommendation System
A hybrid music recommendation engine that uses a combination of real-world chart data, Spotify’s rich metadata, and playlist analysis to provide personalized music suggestions.

## Project Overview (Detailed)
This project integrates data-driven music analysis with real-time API integration to recommend songs that match the user’s preferences, based on two major data sources:

### 1. Billboard Top 100 (Web Scraping)
We extract the latest Billboard Hot 100 chart data using Python-based web scraping tools like BeautifulSoup or requests.

  What’s scraped:
  
    Song title
    
    Artist name
    
    Rank on the chart
  
  Why Billboard?
  
    The Billboard Hot 100 is a reliable benchmark of trending and popular music across genres.
    
    These songs are often used as seed tracks for broader recommendations or to analyze musical trends.
  
  Post-processing:
  
    The scraped tracks are cleaned and formatted.
    
    Track/artist names are passed to the Spotify API to fetch corresponding metadata (track IDs, audio features, etc.).

### Stage 1.5 — Content-Based Song Recommendation Engine

Implemented a cosine-similarity recommendation baseline using nine standardized Spotify-style audio features.

The engine operates on 34,466 unique track records and uses saved preprocessing artifacts to maintain consistent feature transformations.

**Verified functionality:**
- Ranked Top-N song recommendations
- Exclusion of the input track
- Exclusion of matching title–artist releases
- Deduplication of recommended title–artist pairs
- Deterministic ranking
- Input validation and error handling

An initial smoke test using *Comedy* by Gen Hoshino successfully returned ten recommendations with ten distinct track IDs and title–artist pairs.

### Stage 1.6 — Automated Testing

**Environment:** Python 3.11.5, pytest 9.2.0, Windows

**Execution result:** 18 tests passed, 0 failed, in 4.48 seconds.

Tests cover feature-vector validity, dataset alignment, recommendation counts, duplicate prevention, input exclusion, score ordering, determinism and invalid inputs.

**Evaluation limitations:** Functional tests do not establish musical relevance, listener satisfaction or production performance. Multi-query benchmarking and relevance evaluation remain separate validation tasks.

### Stage 1.6 — Recommendation Evaluation

The content-based recommendation engine was evaluated on 100 randomly sampled query tracks using a reproducible random seed of 42. Each query requested ten recommendations.

**Measured results**

| Metric | Result |
|---|---|
| Evaluated queries | 100 |
| Recommendations per query | 10 |
| Mean retrieval latency | 2.81 ms |
| P95 retrieval latency | 3.46 ms |
| Mean cosine similarity | 0.9608 |
| Average exact same-artist recommendations per query | 0.14 |
| Automated tests | 18 passed, 0 failed |

The recommendation engine uses cosine similarity over nine standardized audio features. It excludes the input track and suppresses repeated title–artist combinations.

**Evaluation limitations**

- Similarity is based on audio-feature proximity, not observed listener preferences.
- No ground-truth relevance labels or user interaction data were available.
- The evaluation measured local Python retrieval latency, excluding network, API and deployment overhead.
- Artist matching uses exact normalized artist strings, which may not capture collaborations or aliases.
- Results were collected from 100 sampled queries and may not represent every region of the catalog.

The results establish a functional and measurable baseline for subsequent FAISS retrieval and clustering-based candidate-generation comparisons.


### 2. Spotify Data (via API)
We use the Spotify Web API via the Spotipy Python client to fetch:

  A. Audio Features for any track:
    Danceability
    
    Energy
    
    Acousticness
    
    Valence (mood)
    
    Tempo
    
    Speechiness
    
    Instrumentalness
  
  These features are quantitative and used for calculating similarity between songs.
  
  B. Playlist Analysis
    Playlists are analyzed using their Spotify Playlist ID.
    
    We fetch all tracks in a playlist, then pull their audio features.
    
    These playlists can be:
    
    User-generated playlists
    
    Curated editorial playlists (e.g. "Chill Hits", "Mood Booster")
    
    Genre-specific (e.g. "Lo-fi Beats", "Classical Essentials")

    Recommendation Logic
The recommender system works in multiple modes:

Mode	Description
  Content-Based	Calculates similarity between seed tracks (Billboard or input) and a broader set of songs using cosine similarity on audio features.
  Playlist-Based	Analyzes the “sound” of a playlist and recommends songs with similar feature vectors.
  Hybrid (WIP)	Combines input preferences with global trends (e.g. popular tracks filtered to user’s taste).

The backend handles vector comparisons between Spotify tracks using scikit-learn’s cosine_similarity or distance metrics.


    
