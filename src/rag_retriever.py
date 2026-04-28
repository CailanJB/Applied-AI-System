"""
Phase 4: RAG Retrieval

Takes the query embedding produced by Phase 3 and retrieves the top-k most
semantically similar songs from the FAISS vector index.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from typing import Dict, List, Tuple

import faiss
from embedding_pipeline import query_vector_db


# ---------------------------------------------------------------------------
# Function 1: retrieve_candidates
# ---------------------------------------------------------------------------

def retrieve_candidates(
    query_embedding: np.ndarray,
    index: faiss.IndexFlatL2,
    songs: List[Dict],
    k: int = 20,
) -> List[Tuple[Dict, float]]:
    """Retrieve the k most semantically similar songs from the FAISS index.

    Wraps query_vector_db() with a default candidate set size of 20 and
    guards against requesting more results than the index contains.

    Args:
        query_embedding: Shape (384,) or (1, 384) float32 — output of build_query_embedding().
        index: FAISS IndexFlatL2 built by create_vector_db() in embedding_pipeline.py.
        songs: List of song dicts in the same order as the index.
        k: Number of candidate songs to return (default 20).

    Returns:
        List of (song_dict, l2_distance) tuples, sorted ascending by distance (closest first).
    """
    k = min(k, index.ntotal)
    return query_vector_db(query_embedding, index, songs, k)
