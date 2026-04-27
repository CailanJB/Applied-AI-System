import pytest
from src.query_processor import parse_user_input, extract_playlist_id, match_tracks_to_csv


# ---------------------------------------------------------------------------
# parse_user_input
# ---------------------------------------------------------------------------

def test_parse_detects_playlist_url():
    text = "give me more upbeat tracks from this playlist https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"
    result = parse_user_input(text)

    assert result["input_type"] == "playlist_url"
    assert result["playlist_id"] == "37i9dQZF1DXcBWIGoYBM5M"
    assert result["playlist_url"] == "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"
    assert result["context_text"] == "give me more upbeat tracks from this playlist"


def test_parse_handles_url_with_si_param():
    text = "chill vibes https://open.spotify.com/playlist/abc123?si=xyz789"
    result = parse_user_input(text)

    assert result["input_type"] == "playlist_url"
    assert result["playlist_id"] == "abc123"
    assert result["context_text"] == "chill vibes"


def test_parse_natural_language_only():
    text = "recommend chill songs for studying"
    result = parse_user_input(text)

    assert result["input_type"] == "natural_language"
    assert result["playlist_url"] is None
    assert result["playlist_id"] is None
    assert result["context_text"] == "recommend chill songs for studying"


def test_parse_url_only_no_context():
    text = "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"
    result = parse_user_input(text)

    assert result["input_type"] == "playlist_url"
    assert result["context_text"] == ""


# ---------------------------------------------------------------------------
# extract_playlist_id
# ---------------------------------------------------------------------------

def test_extract_id_plain_url():
    url = "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"
    assert extract_playlist_id(url) == "37i9dQZF1DXcBWIGoYBM5M"


def test_extract_id_with_query_params():
    url = "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M?si=abc123&context=xyz"
    assert extract_playlist_id(url) == "37i9dQZF1DXcBWIGoYBM5M"


def test_extract_id_invalid_url_raises():
    with pytest.raises(ValueError):
        extract_playlist_id("https://open.spotify.com/track/somethingelse")


def test_extract_id_garbage_raises():
    with pytest.raises(ValueError):
        extract_playlist_id("not a url at all")


# ---------------------------------------------------------------------------
# match_tracks_to_csv
# ---------------------------------------------------------------------------

SAMPLE_SONGS = [
    {"title": "Sunrise City", "artist": "Neon Echo", "genre": "pop", "energy": 0.85},
    {"title": "Midnight Rain", "artist": "Luna Frost", "genre": "indie pop", "energy": 0.60},
    {"title": "Deep Roots",   "artist": "Oak & Vine",  "genre": "folk", "energy": 0.40},
]


def test_match_finds_exact_matches():
    playlist_tracks = [
        {"title": "Sunrise City", "artist": "Neon Echo", "album": "Dawn", "duration_ms": 200000},
    ]
    matched, unmatched = match_tracks_to_csv(playlist_tracks, SAMPLE_SONGS)

    assert len(matched) == 1
    assert matched[0]["genre"] == "pop"
    assert len(unmatched) == 0


def test_match_is_case_insensitive():
    playlist_tracks = [
        {"title": "sunrise city", "artist": "NEON ECHO", "album": "Dawn", "duration_ms": 200000},
    ]
    matched, unmatched = match_tracks_to_csv(playlist_tracks, SAMPLE_SONGS)

    assert len(matched) == 1
    assert len(unmatched) == 0


def test_match_returns_unmatched_when_not_in_csv():
    playlist_tracks = [
        {"title": "Unknown Song", "artist": "Ghost Artist", "album": "???", "duration_ms": 180000},
    ]
    matched, unmatched = match_tracks_to_csv(playlist_tracks, SAMPLE_SONGS)

    assert len(matched) == 0
    assert len(unmatched) == 1
    assert unmatched[0]["title"] == "Unknown Song"


def test_match_splits_matched_and_unmatched():
    playlist_tracks = [
        {"title": "Sunrise City", "artist": "Neon Echo",   "album": "Dawn", "duration_ms": 200000},
        {"title": "Not In CSV",   "artist": "Nobody",      "album": "???",  "duration_ms": 180000},
        {"title": "Midnight Rain","artist": "Luna Frost",  "album": "Dusk", "duration_ms": 210000},
    ]
    matched, unmatched = match_tracks_to_csv(playlist_tracks, SAMPLE_SONGS)

    assert len(matched) == 2
    assert len(unmatched) == 1
    assert unmatched[0]["title"] == "Not In CSV"
