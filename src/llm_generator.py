"""
Phase 6: LLM Recommendation Generation

Takes the top re-ranked songs from Phase 5 and uses a Groq-hosted LLM to produce:
  - A creative playlist title
  - A 2-3 sentence overall summary
  - Per-song personalized explanations

Also looks up Spotify track URLs and audio preview links via spotipy
(gracefully skipped if credentials are absent or the search fails).
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from groq import Groq

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    _SPOTIPY_AVAILABLE = True
except ImportError:
    _SPOTIPY_AVAILABLE = False


_SYSTEM_PROMPT = (
    "You are a music curator assistant. Given a user's query, their taste preferences, "
    "and a ranked list of candidate songs with scoring explanations, you will generate "
    "a personalized playlist response. You MUST respond with valid JSON only — no "
    "markdown fences, no preamble, no trailing text."
)

_JSON_SCHEMA = """{
  "playlist_title": "Short evocative name (5-8 words)",
  "summary": "2-3 sentence paragraph describing the playlist and why it fits the user.",
  "songs": [
    {
      "title": "exact song title as given",
      "artist": "exact artist name as given",
      "explanation": "1-2 sentence personalized reason connecting this song to the user's query and preferences"
    }
  ]
}"""

_VALIDATION_SYSTEM_PROMPT = (
    "You are a query validator for a music recommendation system. "
    "Determine if the user's input is a meaningful request for music, songs, or playlists. "
    "A valid query asks for music based on some criteria — mood, activity, genre, artist, "
    "lyrical theme, or similar. "
    "An invalid query contains gibberish, random characters, is completely off-topic, "
    "or has no meaningful music criteria. "
    "Reply with exactly one word: 'valid' or 'invalid'."
)


# ---------------------------------------------------------------------------
# Function 0: validate_query
# ---------------------------------------------------------------------------

def validate_query(
    text: str,
    groq_api_key: Optional[str] = None,
    model: str = "llama3-8b-8192",
) -> Tuple[bool, str]:
    """Check whether a query is a meaningful music request.

    Args:
        text: The raw user query string.
        groq_api_key: API key (falls back to GROQ_API_KEY env var).
        model: Groq model to use — smaller/faster than the generation model.

    Returns:
        (True, "") if valid.
        (False, error_message) if invalid.
    """
    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=5,
        messages=[
            {"role": "system", "content": _VALIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    answer = response.choices[0].message.content.strip().lower()
    if answer.startswith("valid"):
        return True, ""
    return (
        False,
        "Your query doesn't appear to be a meaningful music request. "
        "Try describing a mood, activity, genre, or artist — for example: "
        "'upbeat songs for a morning run' or 'chill jazz for studying'.",
    )


# ---------------------------------------------------------------------------
# Function 1: build_prompt_context
# ---------------------------------------------------------------------------

def build_prompt_context(
    context_text: str,
    user_prefs: Dict,
    results: List[Tuple[Dict, float, str]],
) -> str:
    """Build the user message sent to the LLM.

    Args:
        context_text: The raw user query (from parsed_input["context_text"]).
        user_prefs: Dict with keys genre, mood, energy, acousticness, danceability.
        results: Phase 5 output — list of (song_dict, score, explanation) tuples.

    Returns:
        Formatted multi-section string ready to send as the user message.
    """
    lines = []

    lines.append("## User Query")
    lines.append(context_text or "(no query text provided)")
    lines.append("")

    lines.append("## User Taste Preferences")
    lines.append(f"- Favorite genre: {user_prefs.get('genre', 'any')}")
    lines.append(f"- Favorite mood: {user_prefs.get('mood', 'any')}")
    lines.append(f"- Energy (0=calm, 1=energetic): {user_prefs.get('energy', 0.5):.2f}")
    lines.append(f"- Acousticness (0=electronic, 1=acoustic): {user_prefs.get('acousticness', 0.5):.2f}")
    lines.append(f"- Danceability (0=chill, 1=very danceable): {user_prefs.get('danceability', 0.5):.2f}")
    lines.append("")

    lines.append(f"## Top {len(results)} Candidate Songs (ranked by preference score)")
    for rank, (song, score, explanation) in enumerate(results, 1):
        lines.append(
            f"{rank}. \"{song.get('title', '')}\" by {song.get('artist', '')}"
        )
        lines.append(
            f"   Album: {song.get('album', '')} ({song.get('release_year', '')})"
        )
        lines.append(
            f"   Genre: {song.get('genre', '')} | Mood: {song.get('mood', '')}"
        )
        lines.append(
            f"   Energy: {song.get('energy', 0):.2f} | "
            f"Acousticness: {song.get('acousticness', 0):.2f} | "
            f"Danceability: {song.get('danceability', 0):.2f}"
        )
        lines.append(
            f"   Valence: {song.get('valence', 0):.2f} | "
            f"Tempo: {song.get('tempo_bpm', '')} BPM | "
            f"Popularity: {song.get('popularity', '')}"
        )
        if song.get("description"):
            lines.append(f"   Description: {song['description']}")
        lines.append(f"   Match score: {score:.1f}/10")
        lines.append(f"   Score explanation: {explanation}")
        lines.append("")

    lines.append("## Task")
    lines.append(
        "Generate a JSON response with this exact schema "
        "(one entry per song, in the same order as the ranked list above):"
    )
    lines.append(_JSON_SCHEMA)
    lines.append("Respond with the JSON object only.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Function 2: lookup_spotify_links
# ---------------------------------------------------------------------------

def lookup_spotify_links(
    title: str,
    artist: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Search Spotify for a track and return its URL and preview URL.

    Args:
        title: Song title.
        artist: Artist name.
        client_id: Spotify client ID (falls back to SPOTIFY_CLIENT_ID env var).
        client_secret: Spotify client secret (falls back to SPOTIFY_CLIENT_SECRET env var).

    Returns:
        (spotify_url, preview_url) — both None if credentials are missing or lookup fails.
    """
    try:
        if not _SPOTIPY_AVAILABLE:
            return None, None

        cid = client_id or os.environ.get("SPOTIFY_CLIENT_ID")
        secret = client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET")
        if not cid or not secret:
            return None, None

        auth = SpotifyClientCredentials(client_id=cid, client_secret=secret)
        sp = spotipy.Spotify(auth_manager=auth)

        results = sp.search(q=f"track:{title} artist:{artist}", type="track", limit=1)
        items = results.get("tracks", {}).get("items", [])
        if not items:
            return None, None

        track = items[0]
        spotify_url = track.get("external_urls", {}).get("spotify")
        preview_url = track.get("preview_url")
        return spotify_url, preview_url

    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Function 3: generate_recommendations
# ---------------------------------------------------------------------------

def generate_recommendations(
    context_text: str,
    user_prefs: Dict,
    results: List[Tuple[Dict, float, str]],
    model: str = "llama-3.3-70b-versatile",
    groq_api_key: Optional[str] = None,
) -> Dict:
    """Call a Groq-hosted LLM to generate a playlist title, summary, and per-song explanations.

    Also enriches each song with Spotify URL and preview URL via lookup_spotify_links.

    Args:
        context_text: Raw user query string.
        user_prefs: User taste preferences dict.
        results: Phase 5 output — list of (song_dict, score, explanation).
        model: Groq model ID to use (default: llama-3.3-70b-versatile).
        groq_api_key: API key (falls back to GROQ_API_KEY env var).

    Returns:
        Dict with keys: playlist_title, summary, songs (list of enriched song entries).
    """
    prompt = build_prompt_context(context_text, user_prefs, results)

    client = Groq(api_key=groq_api_key)
    chat_completion = client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    raw_text = chat_completion.choices[0].message.content.strip()

    # Strip markdown fences in case the model adds them despite the system prompt
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[-1]
        if raw_text.endswith("```"):
            raw_text = raw_text[: raw_text.rfind("```")]

    llm_data = json.loads(raw_text)

    # Merge LLM output with Phase 5 results by index position
    enriched_songs = []
    llm_songs = llm_data.get("songs", [])
    for i, (song_dict, score, _) in enumerate(results):
        llm_entry = llm_songs[i] if i < len(llm_songs) else {}
        spotify_url, preview_url = lookup_spotify_links(
            song_dict.get("title", ""),
            song_dict.get("artist", ""),
        )
        enriched_songs.append(
            {
                "title": llm_entry.get("title", song_dict.get("title", "")),
                "artist": llm_entry.get("artist", song_dict.get("artist", "")),
                "explanation": llm_entry.get("explanation", ""),
                "score": score,
                "song_dict": song_dict,
                "spotify_url": spotify_url,
                "preview_url": preview_url,
            }
        )

    return {
        "playlist_title": llm_data.get("playlist_title", "Your Playlist"),
        "summary": llm_data.get("summary", ""),
        "songs": enriched_songs,
    }
