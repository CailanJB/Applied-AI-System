import pytest
from src.reranker import rerank_candidates

# Minimal song dicts — only the fields score_song() reads
SONG_POP_HAPPY = {
    "title": "Sunrise City", "artist": "Neon Echo",
    "genre": "pop", "mood": "happy",
    "energy": 0.85, "acousticness": 0.20, "danceability": 0.80,
}
SONG_ROCK_INTENSE = {
    "title": "Thunder Road", "artist": "Iron Wolves",
    "genre": "rock", "mood": "intense",
    "energy": 0.90, "acousticness": 0.10, "danceability": 0.45,
}
SONG_FOLK_CHILL = {
    "title": "River Stones", "artist": "Oak & Vine",
    "genre": "folk", "mood": "chill",
    "energy": 0.30, "acousticness": 0.80, "danceability": 0.40,
}

# Candidates as Phase 4 would return them: (song_dict, l2_distance)
CANDIDATES = [
    (SONG_POP_HAPPY,   0.25),
    (SONG_ROCK_INTENSE, 0.40),
    (SONG_FOLK_CHILL,  0.60),
]

USER_PREFS_POP = {
    "genre": "pop", "mood": "happy",
    "energy": 0.85, "acousticness": 0.20, "danceability": 0.80,
}


def test_returns_list_of_tuples():
    results = rerank_candidates(CANDIDATES, USER_PREFS_POP)
    assert isinstance(results, list)
    for item in results:
        assert len(item) == 3


def test_results_contain_song_score_explanation():
    results = rerank_candidates(CANDIDATES, USER_PREFS_POP)
    song, score, explanation = results[0]
    assert isinstance(song, dict)
    assert isinstance(score, float)
    assert isinstance(explanation, str)


def test_sorted_by_score_descending():
    results = rerank_candidates(CANDIDATES, USER_PREFS_POP)
    scores = [score for _, score, _ in results]
    assert scores == sorted(scores, reverse=True)


def test_genre_and_mood_match_ranks_first():
    results = rerank_candidates(CANDIDATES, USER_PREFS_POP)
    top_song, _, _ = results[0]
    assert top_song["genre"] == "pop"
    assert top_song["mood"] == "happy"


def test_k_truncates_results():
    results = rerank_candidates(CANDIDATES, USER_PREFS_POP, k=2)
    assert len(results) == 2


def test_k_larger_than_candidates_returns_all():
    results = rerank_candidates(CANDIDATES, USER_PREFS_POP, k=100)
    assert len(results) == len(CANDIDATES)


def test_scores_are_in_valid_range():
    results = rerank_candidates(CANDIDATES, USER_PREFS_POP)
    for _, score, _ in results:
        assert 0.0 <= score <= 10.0


def test_empty_candidates_returns_empty():
    results = rerank_candidates([], USER_PREFS_POP)
    assert results == []
