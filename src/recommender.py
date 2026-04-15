from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import csv
from pathlib import Path

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    songs: List[Dict] = []
    path = Path(csv_path)

    # Resolve relative paths from project root so calls like "data/songs.csv" work
    # whether execution starts at project root or inside src.
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path

    with open(path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            songs.append(
                {
                    "id": int(row["id"]),
                    "title": row["title"],
                    "artist": row["artist"],
                    "genre": row["genre"],
                    "mood": row["mood"],
                    "energy": float(row["energy"]),
                    "tempo_bpm": float(row["tempo_bpm"]),
                    "valence": float(row["valence"]),
                    "danceability": float(row["danceability"]),
                    "acousticness": float(row["acousticness"]),
                }
            )

    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, str]:
    """
    Scores one song against a user's preferences on a 0-10 scale.

    Weights:
    - genre exact match: 3.0
    - mood exact match: 2.0
    - energy closeness: up to 2.5
    - acousticness closeness: up to 1.5
    - danceability closeness: up to 1.0
    """
    score = 0.0
    reasons: List[str] = []

    # Categorical exact-match points.
    if str(song.get("genre", "")).lower() == str(user_prefs.get("genre", "")).lower():
        score += 3.0
        reasons.append("genre match (+3.0)")
    else:
        reasons.append("genre no match (+0.0)")

    if str(song.get("mood", "")).lower() == str(user_prefs.get("mood", "")).lower():
        score += 2.0
        reasons.append("mood match (+2.0)")
    else:
        reasons.append("mood no match (+0.0)")

    # Numeric closeness points: points = weight * max(0, 1 - |song - user|)
    energy_sim = max(0.0, 1.0 - abs(float(song.get("energy", 0.0)) - float(user_prefs.get("energy", 0.0))))
    energy_points = 2.5 * energy_sim
    score += energy_points
    reasons.append(f"energy closeness (+{energy_points:.2f})")

    acoustic_sim = max(
        0.0,
        1.0 - abs(float(song.get("acousticness", 0.0)) - float(user_prefs.get("acousticness", 0.0))),
    )
    acoustic_points = 1.5 * acoustic_sim
    score += acoustic_points
    reasons.append(f"acousticness closeness (+{acoustic_points:.2f})")

    dance_sim = max(
        0.0,
        1.0 - abs(float(song.get("danceability", 0.0)) - float(user_prefs.get("danceability", 0.0))),
    )
    dance_points = 1.0 * dance_sim
    score += dance_points
    reasons.append(f"danceability closeness (+{dance_points:.2f})")

    return round(score, 2), "; ".join(reasons)

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    if k <= 0 or not songs:
        return []

    scored_songs: List[Tuple[Dict, float, str]] = []
    for song in songs:
        score, explanation = score_song(user_prefs, song)
        scored_songs.append((song, score, explanation))

    scored_songs.sort(key=lambda item: item[1], reverse=True)
    return scored_songs[:k]
