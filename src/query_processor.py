"""
Phase 2: User Input & Query Processing

Accepts raw user text and converts it into structured query data
ready for Phase 3 embedding. Handles two input formats:
  - context + Spotify playlist URL
  - natural language only
"""

import os
import re
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

load_dotenv()

# Matches https://open.spotify.com/playlist/<ID> with optional ?si=... or other params
_SPOTIFY_PLAYLIST_RE = re.compile(
    r"https://open\.spotify\.com/playlist/([a-zA-Z0-9]+)(?:\?[^\s]*)?"
)


# ---------------------------------------------------------------------------
# Function 1: parse_user_input
# ---------------------------------------------------------------------------

def parse_user_input(text: str) -> Dict:
    """Detect input type and split URL from surrounding context text.

    Returns a dict with keys:
        input_type   : "playlist_url" | "natural_language"
        playlist_url : full matched URL string, or None
        playlist_id  : extracted Spotify playlist ID, or None
        context_text : the non-URL portion of the input (stripped)
    """
    match = _SPOTIFY_PLAYLIST_RE.search(text)

    if match:
        full_url = match.group(0)
        playlist_id = match.group(1)
        context_text = _SPOTIFY_PLAYLIST_RE.sub("", text).strip()
        return {
            "input_type": "playlist_url",
            "playlist_url": full_url,
            "playlist_id": playlist_id,
            "context_text": context_text,
        }

    return {
        "input_type": "natural_language",
        "playlist_url": None,
        "playlist_id": None,
        "context_text": text.strip(),
    }


# ---------------------------------------------------------------------------
# Function 2: extract_playlist_id
# ---------------------------------------------------------------------------

def extract_playlist_id(url: str) -> str:
    """Extract the alphanumeric playlist ID from a Spotify playlist URL.

    Handles:
      https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M
      https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M?si=abc123

    Raises ValueError if the URL is not a valid Spotify playlist URL.
    """
    match = _SPOTIFY_PLAYLIST_RE.search(url)
    if not match:
        raise ValueError(f"Not a valid Spotify playlist URL: {url!r}")
    return match.group(1)


# ---------------------------------------------------------------------------
# Function 3: fetch_playlist_tracks
# ---------------------------------------------------------------------------

"""for tracks not in the csv we create new song objects for them via this function"""
def _infer_mood(valence: float, energy: float) -> str:
    """Infer a mood label from Spotify's valence and energy values."""
    if valence > 0.7 and energy > 0.7:
        return "happy"
    if valence > 0.7 and energy > 0.4:
        return "peaceful"
    if valence > 0.7:
        return "relaxed"
    if valence > 0.4 and energy > 0.7:
        return "optimistic"
    if valence > 0.4 and energy > 0.4:
        return "chill"
    if valence > 0.4:
        return "focused"
    if energy > 0.7:
        return "intense"
    if energy > 0.4:
        return "moody"
    return "melancholic"


def fetch_playlist_tracks(
    playlist_id: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> List[Dict]:
    """Fetch all tracks from a public Spotify playlist, enriched with audio attributes.

    Credentials are read from SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET
    environment variables if not passed directly.

    Returns a list of fully enriched track dicts matching the CSV song schema:
        { title, artist, album, genre, mood,
          energy, tempo_bpm, valence, danceability, acousticness,
          lyrics, description, release_year, popularity }
    """
    cid = client_id or os.environ.get("SPOTIFY_CLIENT_ID")
    secret = client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET")

    if not cid or not secret:
        raise EnvironmentError(
            "Spotify credentials missing. Set SPOTIFY_CLIENT_ID and "
            "SPOTIFY_CLIENT_SECRET in your .env file or pass them directly."
        )

    auth_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    # Pass 1: collect raw track stubs and IDs from all playlist pages
    raw_tracks: List[Dict] = []
    response = sp.playlist_tracks(playlist_id)
    while response:
        for item in response["items"]:
            track = item.get("track")
            if not track:
                continue
            artist = track["artists"][0] if track["artists"] else {}
            raw_tracks.append({
                "track_id":     track["id"],
                "artist_id":    artist.get("id", ""),
                "title":        track["name"],
                "artist":       artist.get("name", ""),
                "album":        track["album"]["name"] if track.get("album") else "",
                "release_date": track["album"].get("release_date", "") if track.get("album") else "",
                "popularity":   track.get("popularity", 0),
            })
        response = sp.next(response) if response.get("next") else None

    # Pass 2: batch-fetch audio features (max 100 per call)
    track_ids = [t["track_id"] for t in raw_tracks if t["track_id"]]
    audio_features: Dict[str, Dict] = {}
    for i in range(0, len(track_ids), 100):
        batch = sp.audio_features(track_ids[i:i + 100]) or []
        for feat in batch:
            if feat:
                audio_features[feat["id"]] = feat

    # Pass 3: batch-fetch artist genres (max 50 per call)
    artist_ids = list({t["artist_id"] for t in raw_tracks if t["artist_id"]})
    artist_genres: Dict[str, str] = {}
    for i in range(0, len(artist_ids), 50):
        result = sp.artists(artist_ids[i:i + 50])
        for artist in result.get("artists") or []:
            if artist:
                genres = artist.get("genres") or []
                artist_genres[artist["id"]] = genres[0] if genres else "unknown"

    # Pass 4: assemble enriched track dicts
    enriched: List[Dict] = []
    for raw in raw_tracks:
        feat = audio_features.get(raw["track_id"], {})
        valence = feat.get("valence", 0.5)
        energy  = feat.get("energy",  0.5)
        genre   = artist_genres.get(raw["artist_id"], "unknown")
        mood    = _infer_mood(valence, energy)
        release_date = raw.get("release_date", "")
        enriched.append({
            "title":        raw["title"],
            "artist":       raw["artist"],
            "album":        raw["album"],
            "genre":        genre,
            "mood":         mood,
            "energy":       round(energy, 4),
            "tempo_bpm":    feat.get("tempo", 0.0),
            "valence":      round(valence, 4),
            "danceability": round(feat.get("danceability", 0.5), 4),
            "acousticness": round(feat.get("acousticness", 0.5), 4),
            "lyrics":       "",
            "description":  f"A {mood} {genre} track by {raw['artist']}.",
            "release_year": int(release_date[:4]) if release_date else 0,
            "popularity":   raw.get("popularity", 0),
        })

    return enriched


# ---------------------------------------------------------------------------
# Function 4: match_tracks_to_csv
# ---------------------------------------------------------------------------

def match_tracks_to_csv(
    playlist_tracks: List[Dict],
    songs: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """Match Spotify tracks against the local CSV song catalogue.

    Matched songs return their full CSV dict (which has pre-computed embeddings).
    Unmatched tracks are returned as-is for on-the-fly embedding in Phase 3.

    Matching: case-insensitive exact match on title + artist.

    Returns:
    (matched_csv_songs, unmatched_spotify_tracks)
    """
    # Build a lookup key from CSV songs: (title_lower, artist_lower) -> song dict
    csv_index: Dict[Tuple[str, str], Dict] = {
        (s["title"].lower(), s["artist"].lower()): s
        for s in songs
    }

    matched: List[Dict] = []
    unmatched: List[Dict] = []

    for track in playlist_tracks:
        key = (track["title"].lower(), track["artist"].lower())
        csv_song = csv_index.get(key)
        if csv_song:
            matched.append(csv_song)
        else:
            unmatched.append(track)

    return matched, unmatched
