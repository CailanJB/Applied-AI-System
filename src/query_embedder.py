"""
Phase 3: Query Embedding Creation

Converts parsed user input into a single 384-dim float32 query embedding
ready to be passed into query_vector_db() (Phase 4).

Two paths:
  - natural_language : encode context text directly
  - playlist_url     : build playlist central vector, blend 70% playlist / 30% context
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer

from embedding_pipeline import song_to_text


# ---------------------------------------------------------------------------
# Function 1: load_embedding_model
# ---------------------------------------------------------------------------

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load and return the SentenceTransformer model.

    Load once and pass the returned model into all other functions.
    Must use the same model that built the knowledge base (all-MiniLM-L6-v2, 384-dim).
    """
    return SentenceTransformer(model_name)


# ---------------------------------------------------------------------------
# Function 2: embed_text
# ---------------------------------------------------------------------------

"""Takes in text string and encoded into vector embedding"""
def embed_text(text: str, model: SentenceTransformer) -> np.ndarray:
    """Encode a single string into a (384,) float32 numpy array.

    Used for natural language queries and the context portion of playlist inputs.
    Empty strings produce a near-zero vector, which is handled gracefully by blend_embeddings.
    """
    vector = model.encode(text, convert_to_numpy=True).astype("float32")
    return vector  # shape (384,)


# ---------------------------------------------------------------------------
# Function 3: embed_playlist_songs
# ---------------------------------------------------------------------------

"""Matched csv songs result returned from query processing function
                    """
def embed_playlist_songs(
    matched_csv_songs: List[Dict],
    unmatched_tracks: List[Dict],
    model: SentenceTransformer,
) -> np.ndarray:
    """Build the playlist central vector — a single embedding representing
    the overall vibe of the playlist.

    Matched CSV songs are encoded via song_to_text() (full metadata).
    Unmatched Spotify tracks are encoded from title + artist only.
    All embeddings are averaged into one (384,) float32 vector.

    Raises ValueError if both lists are empty.
    """
    if not matched_csv_songs and not unmatched_tracks:
        raise ValueError("Cannot embed an empty playlist: both matched and unmatched lists are empty.")

    texts: List[str] = []

    for song in matched_csv_songs:
        texts.append(song_to_text(song))

    for track in unmatched_tracks:
        texts.append(f"Title: {track['title']}, Artist: {track['artist']}")

    embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")  # (N, 384)
    central_vector = embeddings.mean(axis=0)  # (384,)
    return central_vector


# ---------------------------------------------------------------------------
# Function 4: blend_embeddings
# ---------------------------------------------------------------------------

def blend_embeddings(
    playlist_vector: np.ndarray,
    context_vector: np.ndarray,
    playlist_weight: float = 0.7,
) -> np.ndarray:
    """Blend the playlist central vector with the context text vector.

    blended = playlist_weight * playlist_vector + (1 - playlist_weight) * context_vector

    L2-normalizes the result so FAISS distance comparisons are consistent.
    Returns shape (384,) float32.
    """
    context_weight = 1.0 - playlist_weight
    blended = playlist_weight * playlist_vector + context_weight * context_vector

    norm = np.linalg.norm(blended)
    if norm > 0:
        blended = blended / norm

    return blended.astype("float32")


# ---------------------------------------------------------------------------
# Function 5: build_query_embedding
# ---------------------------------------------------------------------------

def build_query_embedding(
    parsed_input: Dict,
    matched_csv_songs: List[Dict],
    unmatched_tracks: List[Dict],
    model: SentenceTransformer,
) -> np.ndarray:
    """Top-level orchestrator: convert parsed user input into a query embedding.

    Natural language path:
        embed_text(context_text, model)

    Playlist URL path:
        embed_playlist_songs(matched, unmatched, model)  -> playlist_vec
        embed_text(context_text, model)                  -> context_vec
        blend_embeddings(playlist_vec, context_vec)      -> final (384,)

    Returns shape (384,) float32, ready for query_vector_db().
    """
    input_type = parsed_input["input_type"]
    context_text = parsed_input.get("context_text", "")

    if input_type == "natural_language":
        return embed_text(context_text, model)

    # playlist_url path
    playlist_vec = embed_playlist_songs(matched_csv_songs, unmatched_tracks, model)
    context_vec = embed_text(context_text, model)
    return blend_embeddings(playlist_vec, context_vec)
