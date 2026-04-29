import json
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from llm_generator import build_prompt_context, generate_recommendations, lookup_spotify_links, validate_query

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MINIMAL_PREFS = {
    "genre": "pop",
    "mood": "happy",
    "energy": 0.8,
    "acousticness": 0.2,
    "danceability": 0.7,
}

SONG_A = {
    "title": "Sunrise City",
    "artist": "Neon Echo",
    "album": "Urban Sunrise",
    "genre": "pop",
    "mood": "happy",
    "energy": 0.82,
    "acousticness": 0.18,
    "danceability": 0.79,
    "valence": 0.84,
    "tempo_bpm": 118,
    "description": "An uplifting pop song.",
    "release_year": 2022,
    "popularity": 90,
}

SONG_B = {
    "title": "Thunder Road",
    "artist": "Iron Wolves",
    "album": "Roar",
    "genre": "rock",
    "mood": "intense",
    "energy": 0.90,
    "acousticness": 0.10,
    "danceability": 0.45,
    "valence": 0.60,
    "tempo_bpm": 140,
    "description": "Heavy rock anthem.",
    "release_year": 2019,
    "popularity": 75,
}

MINIMAL_RESULTS = [
    (SONG_A, 8.5, "genre match (+3.0); mood match (+2.0); energy closeness (+2.45)"),
]

TWO_RESULTS = [
    (SONG_A, 8.5, "genre match (+3.0); mood match (+2.0)"),
    (SONG_B, 5.1, "energy closeness (+2.10)"),
]

FAKE_LLM_JSON_ONE = {
    "playlist_title": "Test Playlist",
    "summary": "A great test playlist. It fits your vibe perfectly. Enjoy.",
    "songs": [
        {
            "title": "Sunrise City",
            "artist": "Neon Echo",
            "explanation": "This upbeat pop track matches your love of happy, energetic music.",
        }
    ],
}

FAKE_LLM_JSON_TWO = {
    "playlist_title": "Double Feature",
    "summary": "Two contrasting tracks. Enjoy the variety.",
    "songs": [
        {
            "title": "Sunrise City",
            "artist": "Neon Echo",
            "explanation": "Uplifting pop that matches your mood perfectly.",
        },
        {
            "title": "Thunder Road",
            "artist": "Iron Wolves",
            "explanation": "A high-energy rock track to keep things interesting.",
        },
    ],
}


def _make_mock_groq(llm_json: dict):
    """Build a mock Groq client that returns llm_json as the chat completion content."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(llm_json)
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


# ---------------------------------------------------------------------------
# Group 1: build_prompt_context — pure unit tests, no mocks
# ---------------------------------------------------------------------------

def test_build_prompt_context_returns_string():
    result = build_prompt_context("chill late night vibes", MINIMAL_PREFS, MINIMAL_RESULTS)
    assert isinstance(result, str)


def test_build_prompt_context_contains_query():
    prompt = build_prompt_context("chill late night vibes", MINIMAL_PREFS, MINIMAL_RESULTS)
    assert "chill late night vibes" in prompt


def test_build_prompt_context_contains_song_title():
    prompt = build_prompt_context("any query", MINIMAL_PREFS, MINIMAL_RESULTS)
    assert "Sunrise City" in prompt


def test_build_prompt_context_contains_artist():
    prompt = build_prompt_context("any query", MINIMAL_PREFS, MINIMAL_RESULTS)
    assert "Neon Echo" in prompt


def test_build_prompt_context_contains_user_genre():
    prompt = build_prompt_context("any query", MINIMAL_PREFS, MINIMAL_RESULTS)
    assert "pop" in prompt


def test_build_prompt_context_all_songs_included():
    ten_results = MINIMAL_RESULTS * 10
    prompt = build_prompt_context("q", MINIMAL_PREFS, ten_results)
    assert prompt.count("Sunrise City") == 10


def test_build_prompt_context_score_appears():
    prompt = build_prompt_context("q", MINIMAL_PREFS, MINIMAL_RESULTS)
    assert "8.5" in prompt


# ---------------------------------------------------------------------------
# Group 2: generate_recommendations — mock Groq client
# ---------------------------------------------------------------------------

@patch("llm_generator.lookup_spotify_links", return_value=(None, None))
@patch("llm_generator.Groq")
def test_generate_returns_expected_keys(mock_groq_cls, mock_spotify):
    mock_groq_cls.return_value = _make_mock_groq(FAKE_LLM_JSON_ONE)
    output = generate_recommendations("test query", MINIMAL_PREFS, MINIMAL_RESULTS)
    assert "playlist_title" in output
    assert "summary" in output
    assert "songs" in output


@patch("llm_generator.lookup_spotify_links", return_value=(None, None))
@patch("llm_generator.Groq")
def test_generate_songs_list_length_matches_results(mock_groq_cls, mock_spotify):
    mock_groq_cls.return_value = _make_mock_groq(FAKE_LLM_JSON_TWO)
    output = generate_recommendations("q", MINIMAL_PREFS, TWO_RESULTS)
    assert len(output["songs"]) == len(TWO_RESULTS)


@patch("llm_generator.lookup_spotify_links", return_value=(None, None))
@patch("llm_generator.Groq")
def test_generate_songs_contain_required_fields(mock_groq_cls, mock_spotify):
    mock_groq_cls.return_value = _make_mock_groq(FAKE_LLM_JSON_ONE)
    output = generate_recommendations("q", MINIMAL_PREFS, MINIMAL_RESULTS)
    entry = output["songs"][0]
    for field in ("title", "artist", "explanation", "score", "song_dict", "spotify_url", "preview_url"):
        assert field in entry, f"Missing field: {field}"


@patch("llm_generator.lookup_spotify_links", return_value=(None, None))
@patch("llm_generator.Groq")
def test_generate_score_forwarded_from_phase5(mock_groq_cls, mock_spotify):
    mock_groq_cls.return_value = _make_mock_groq(FAKE_LLM_JSON_ONE)
    output = generate_recommendations("q", MINIMAL_PREFS, MINIMAL_RESULTS)
    assert output["songs"][0]["score"] == 8.5


@patch("llm_generator.lookup_spotify_links", return_value=(None, None))
@patch("llm_generator.Groq")
def test_generate_calls_spotify_lookup_per_song(mock_groq_cls, mock_spotify):
    mock_groq_cls.return_value = _make_mock_groq(FAKE_LLM_JSON_TWO)
    generate_recommendations("q", MINIMAL_PREFS, TWO_RESULTS)
    assert mock_spotify.call_count == len(TWO_RESULTS)


@patch("llm_generator.lookup_spotify_links", return_value=(None, None))
@patch("llm_generator.Groq")
def test_generate_strips_markdown_fences(mock_groq_cls, mock_spotify):
    fenced = "```json\n" + json.dumps(FAKE_LLM_JSON_ONE) + "\n```"
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = fenced
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    mock_groq_cls.return_value = mock_client

    output = generate_recommendations("q", MINIMAL_PREFS, MINIMAL_RESULTS)
    assert output["playlist_title"] == "Test Playlist"


@patch("llm_generator.lookup_spotify_links", return_value=(None, None))
@patch("llm_generator.Groq")
def test_generate_song_dict_forwarded(mock_groq_cls, mock_spotify):
    mock_groq_cls.return_value = _make_mock_groq(FAKE_LLM_JSON_ONE)
    output = generate_recommendations("q", MINIMAL_PREFS, MINIMAL_RESULTS)
    assert output["songs"][0]["song_dict"] is SONG_A


# ---------------------------------------------------------------------------
# Group 3: lookup_spotify_links — mock spotipy
# ---------------------------------------------------------------------------

def test_lookup_returns_none_when_no_credentials(monkeypatch):
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)
    url, preview = lookup_spotify_links("Any Song", "Any Artist", client_id=None, client_secret=None)
    assert url is None
    assert preview is None


@patch("llm_generator._SPOTIPY_AVAILABLE", False)
def test_lookup_returns_none_when_spotipy_unavailable():
    url, preview = lookup_spotify_links("Song", "Artist", client_id="id", client_secret="secret")
    assert url is None
    assert preview is None


@patch("llm_generator._SPOTIPY_AVAILABLE", True)
@patch("llm_generator.SpotifyClientCredentials")
@patch("llm_generator.spotipy.Spotify")
def test_lookup_returns_urls_when_found(mock_sp_cls, mock_creds):
    mock_sp = MagicMock()
    mock_sp_cls.return_value = mock_sp
    mock_sp.search.return_value = {
        "tracks": {
            "items": [
                {
                    "external_urls": {"spotify": "https://open.spotify.com/track/abc"},
                    "preview_url": "https://p.scdn.co/preview/abc",
                }
            ]
        }
    }
    url, preview = lookup_spotify_links("Sunrise City", "Neon Echo", client_id="id", client_secret="secret")
    assert url == "https://open.spotify.com/track/abc"
    assert preview == "https://p.scdn.co/preview/abc"


@patch("llm_generator._SPOTIPY_AVAILABLE", True)
@patch("llm_generator.SpotifyClientCredentials")
@patch("llm_generator.spotipy.Spotify")
def test_lookup_returns_none_on_empty_results(mock_sp_cls, mock_creds):
    mock_sp = MagicMock()
    mock_sp_cls.return_value = mock_sp
    mock_sp.search.return_value = {"tracks": {"items": []}}
    url, preview = lookup_spotify_links("Ghost Song", "Nobody", client_id="id", client_secret="secret")
    assert url is None
    assert preview is None


@patch("llm_generator._SPOTIPY_AVAILABLE", True)
@patch("llm_generator.SpotifyClientCredentials")
@patch("llm_generator.spotipy.Spotify")
def test_lookup_returns_none_on_api_exception(mock_sp_cls, mock_creds):
    mock_sp = MagicMock()
    mock_sp_cls.return_value = mock_sp
    mock_sp.search.side_effect = Exception("network error")
    url, preview = lookup_spotify_links("Song", "Artist", client_id="id", client_secret="secret")
    assert url is None
    assert preview is None


# ---------------------------------------------------------------------------
# Group 4: validate_query — mock Groq client
# ---------------------------------------------------------------------------

def _make_validation_mock(answer: str):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = answer
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


@patch("llm_generator.Groq")
def test_validate_returns_true_for_valid_query(mock_groq_cls):
    mock_groq_cls.return_value = _make_validation_mock("valid")
    is_valid, reason = validate_query("chill songs for late night studying")
    assert is_valid is True
    assert reason == ""


@patch("llm_generator.Groq")
def test_validate_returns_false_for_garbage(mock_groq_cls):
    mock_groq_cls.return_value = _make_validation_mock("invalid")
    is_valid, reason = validate_query("hshshhaahhshs")
    assert is_valid is False
    assert isinstance(reason, str) and len(reason) > 0


@patch("llm_generator.Groq")
def test_validate_returns_false_for_partial_gibberish(mock_groq_cls):
    mock_groq_cls.return_value = _make_validation_mock("invalid")
    is_valid, reason = validate_query("provide me songs that xgxgscbhcshvsu")
    assert is_valid is False


@patch("llm_generator.Groq")
def test_validate_tolerates_punctuated_answer(mock_groq_cls):
    mock_groq_cls.return_value = _make_validation_mock("valid.")
    is_valid, _ = validate_query("upbeat pop songs")
    assert is_valid is True


@patch("llm_generator.Groq")
def test_validate_uses_small_fast_model(mock_groq_cls):
    mock_groq_cls.return_value = _make_validation_mock("valid")
    validate_query("any query")
    call_kwargs = mock_groq_cls.return_value.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "llama3-8b-8192"
    assert call_kwargs["max_tokens"] == 5
