import numpy as np
import pytest
import faiss
from src.rag_retriever import retrieve_candidates

DIM = 384
N_SONGS = 5


def _make_index_and_songs():
    rng = np.random.default_rng(42)
    embeddings = rng.random((N_SONGS, DIM)).astype(np.float32)
    index = faiss.IndexFlatL2(DIM)
    index.add(embeddings)
    songs = [{"title": f"Song {i}", "artist": f"Artist {i}"} for i in range(N_SONGS)]
    return index, songs


INDEX, SONGS = _make_index_and_songs()
QUERY = np.random.default_rng(0).random(DIM).astype(np.float32)


def test_returns_k_results():
    results = retrieve_candidates(QUERY, INDEX, SONGS, k=3)
    assert len(results) == 3


def test_returns_all_when_k_equals_index_size():
    results = retrieve_candidates(QUERY, INDEX, SONGS, k=N_SONGS)
    assert len(results) == N_SONGS


def test_k_clamped_to_index_size():
    results = retrieve_candidates(QUERY, INDEX, SONGS, k=100)
    assert len(results) == N_SONGS


def test_results_are_song_distance_tuples():
    results = retrieve_candidates(QUERY, INDEX, SONGS, k=3)
    for song, dist in results:
        assert isinstance(song, dict)
        assert isinstance(dist, float)
        assert dist >= 0.0


def test_results_sorted_ascending_by_distance():
    results = retrieve_candidates(QUERY, INDEX, SONGS, k=N_SONGS)
    distances = [dist for _, dist in results]
    assert distances == sorted(distances)


def test_default_k_clamped_to_index():
    results = retrieve_candidates(QUERY, INDEX, SONGS)
    assert len(results) == N_SONGS


def test_accepts_1d_query_embedding():
    vec = np.random.default_rng(7).random(DIM).astype(np.float32)
    results = retrieve_candidates(vec, INDEX, SONGS, k=2)
    assert len(results) == 2
