import numpy as np
import pytest
from src.query_embedder import (
    load_embedding_model,
    embed_text,
    embed_playlist_songs,
    blend_embeddings,
    build_query_embedding,
)

# Load once for the whole test module — model loading is slow
MODEL = load_embedding_model()

SAMPLE_CSV_SONGS = [
    {
        "title": "Sunrise City", "artist": "Neon Echo", "genre": "pop",
        "mood": "happy", "description": "An upbeat pop track", "album": "Dawn",
    },
    {
        "title": "Midnight Rain", "artist": "Luna Frost", "genre": "indie pop",
        "mood": "chill", "description": "A mellow indie track", "album": "Dusk",
    },
]

SAMPLE_UNMATCHED = [
    {"title": "Unknown Banger", "artist": "Ghost DJ", "album": "???", "duration_ms": 180000},
]


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------

def test_embed_text_returns_correct_shape():
    vec = embed_text("chill songs for studying", MODEL)
    assert vec.shape == (384,)
    assert vec.dtype == np.float32


def test_embed_text_empty_string_does_not_crash():
    vec = embed_text("", MODEL)
    assert vec.shape == (384,)
    assert vec.dtype == np.float32


def test_embed_text_different_inputs_produce_different_vectors():
    v1 = embed_text("upbeat dance pop", MODEL)
    v2 = embed_text("sad acoustic folk", MODEL)
    assert not np.allclose(v1, v2)


# ---------------------------------------------------------------------------
# embed_playlist_songs
# ---------------------------------------------------------------------------

def test_embed_playlist_songs_returns_correct_shape():
    vec = embed_playlist_songs(SAMPLE_CSV_SONGS, [], MODEL)
    assert vec.shape == (384,)
    assert vec.dtype == np.float32


def test_embed_playlist_songs_with_unmatched_only():
    vec = embed_playlist_songs([], SAMPLE_UNMATCHED, MODEL)
    assert vec.shape == (384,)


def test_embed_playlist_songs_mixed_matched_and_unmatched():
    vec = embed_playlist_songs(SAMPLE_CSV_SONGS, SAMPLE_UNMATCHED, MODEL)
    assert vec.shape == (384,)


def test_embed_playlist_songs_empty_raises():
    with pytest.raises(ValueError):
        embed_playlist_songs([], [], MODEL)


# ---------------------------------------------------------------------------
# blend_embeddings
# ---------------------------------------------------------------------------

def test_blend_returns_correct_shape():
    v1 = embed_text("upbeat pop", MODEL)
    v2 = embed_text("chill study", MODEL)
    blended = blend_embeddings(v1, v2)
    assert blended.shape == (384,)
    assert blended.dtype == np.float32


def test_blend_is_l2_normalized():
    v1 = embed_text("upbeat pop", MODEL)
    v2 = embed_text("chill study", MODEL)
    blended = blend_embeddings(v1, v2)
    norm = np.linalg.norm(blended)
    assert abs(norm - 1.0) < 1e-5


def test_blend_weight_70_30():
    # With weight=1.0 the result should be very close to the playlist vector (normalized)
    v1 = embed_text("heavy metal", MODEL)
    v2 = np.zeros(384, dtype="float32")  # zero context vector
    blended = blend_embeddings(v1, v2, playlist_weight=1.0)
    v1_norm = v1 / np.linalg.norm(v1)
    assert np.allclose(blended, v1_norm, atol=1e-5)


# ---------------------------------------------------------------------------
# build_query_embedding
# ---------------------------------------------------------------------------

def test_build_natural_language_path():
    parsed = {
        "input_type": "natural_language",
        "playlist_url": None,
        "playlist_id": None,
        "context_text": "recommend chill songs for studying",
    }
    vec = build_query_embedding(parsed, [], [], MODEL)
    assert vec.shape == (384,)
    assert vec.dtype == np.float32


def test_build_playlist_url_path():
    parsed = {
        "input_type": "playlist_url",
        "playlist_url": "https://open.spotify.com/playlist/abc123",
        "playlist_id": "abc123",
        "context_text": "give me more upbeat tracks",
    }
    vec = build_query_embedding(parsed, SAMPLE_CSV_SONGS, SAMPLE_UNMATCHED, MODEL)
    assert vec.shape == (384,)
    assert vec.dtype == np.float32


def test_build_playlist_url_empty_context():
    parsed = {
        "input_type": "playlist_url",
        "playlist_url": "https://open.spotify.com/playlist/abc123",
        "playlist_id": "abc123",
        "context_text": "",
    }
    vec = build_query_embedding(parsed, SAMPLE_CSV_SONGS, [], MODEL)
    assert vec.shape == (384,)


def test_build_natural_language_differs_from_playlist():
    parsed_nl = {
        "input_type": "natural_language",
        "playlist_url": None,
        "playlist_id": None,
        "context_text": "upbeat dance tracks",
    }
    parsed_pl = {
        "input_type": "playlist_url",
        "playlist_url": "https://open.spotify.com/playlist/abc123",
        "playlist_id": "abc123",
        "context_text": "upbeat dance tracks",
    }
    vec_nl = build_query_embedding(parsed_nl, [], [], MODEL)
    vec_pl = build_query_embedding(parsed_pl, SAMPLE_CSV_SONGS, [], MODEL)
    # The two paths with the same context text should produce different vectors
    assert not np.allclose(vec_nl, vec_pl)
