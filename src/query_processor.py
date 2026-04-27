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

def fetch_playlist_tracks(
    playlist_id: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> List[Dict]:
    """Fetch all tracks from a public Spotify playlist.

    Credentials are read from SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET
    environment variables if not passed directly.

    Returns a list of simplified track dicts:
        { title, artist, album, duration_ms }
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

    tracks: List[Dict] = []
    response = sp.playlist_tracks(playlist_id)

    while response:
        for item in response["items"]:
            track = item.get("track")
            if not track:
                continue
            tracks.append({
                "title": track["name"],
                "artist": track["artists"][0]["name"] if track["artists"] else "",
                "album": track["album"]["name"] if track.get("album") else "",
                "duration_ms": track.get("duration_ms", 0),
            })
        response = sp.next(response) if response.get("next") else None

    return tracks


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
