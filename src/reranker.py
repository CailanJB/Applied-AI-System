"""
Phase 5: Re-ranking

Takes the candidate set from Phase 4 (retrieved by semantic similarity) and
re-ranks it using score_song() from recommender.py, which scores each song
against the user's taste preferences (genre, mood, energy, acousticness,
danceability) on a 0–10 scale.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from typing import Dict, List, Tuple

from recommender import score_song


# ---------------------------------------------------------------------------
# Function 1: rerank_candidates
# ---------------------------------------------------------------------------

def rerank_candidates(
    candidates: List[Tuple[Dict, float]],
    user_prefs: Dict,
    k: int = 10,
) -> List[Tuple[Dict, float, str]]:
    """Score and re-rank retrieved candidates by user preference match.

    For each (song, l2_distance) pair from Phase 4, calls score_song() to
    compute a 0–10 preference score and explanation. Results are sorted by
    score descending so the best personal match comes first.

    Args:
        candidates: Output of retrieve_candidates() — list of (song_dict, l2_distance).
        user_prefs: User taste preferences dict with keys:
                    genre (str), mood (str), energy (float),
                    acousticness (float), danceability (float).
        k: Maximum number of results to return (default 10).

    Returns:
        List of (song_dict, score, explanation) tuples, sorted by score descending.
    """
    scored: List[Tuple[Dict, float, str]] = []
    for song, _ in candidates:
        score, explanation = score_song(user_prefs, song)
        scored.append((song, score, explanation))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]
