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


    
