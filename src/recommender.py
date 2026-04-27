from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
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
    lyrics: str
    description: str
    album: str
    release_year: int
    popularity: int

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
        """Stores the song catalog used for recommendations."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Returns the top-k songs ranked by match score for a user profile."""
        if k <= 0 or not self.songs:
            return []

        prefs = _prefs_from_user_profile(user)
        scored_songs = [
            (song, score_song(prefs, asdict(song))[0])
            for song in self.songs
        ]
        scored_songs.sort(key=lambda item: item[1], reverse=True)
        return [song for song, _ in scored_songs[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Generates a human-readable explanation for a song's score."""
        prefs = _prefs_from_user_profile(user)
        _, explanation = score_song(prefs, asdict(song))
        return explanation

def _prefs_from_user_profile(user: UserProfile) -> Dict:
    """Converts a UserProfile into the dict format expected by score_song."""
    return {
        "genre": user.favorite_genre,
        "mood": user.favorite_mood,
        "energy": user.target_energy,
        "acousticness": 0.8 if user.likes_acoustic else 0.2,
        "danceability": 0.6,
    }

def _song_from_dict(song_dict: Dict) -> Song:
    """Builds a Song dataclass instance from a song dictionary."""
    return Song(
        id=int(song_dict["id"]),
        title=str(song_dict["title"]),
        artist=str(song_dict["artist"]),
        genre=str(song_dict["genre"]),
        mood=str(song_dict["mood"]),
        energy=float(song_dict["energy"]),
        tempo_bpm=float(song_dict["tempo_bpm"]),
        valence=float(song_dict["valence"]),
        danceability=float(song_dict["danceability"]),
        acousticness=float(song_dict["acousticness"]),
        lyrics=str(song_dict["lyrics"]),
        description=str(song_dict["description"]),
        album=str(song_dict["album"]),
        release_year=int(song_dict["release_year"]),
        popularity=int(song_dict["popularity"]),
    )

def _user_profile_from_prefs(user_prefs: Dict) -> UserProfile:
    """Converts functional preference data into a UserProfile instance."""
    acoustic_pref = user_prefs.get("acousticness", 0.5)
    if isinstance(acoustic_pref, bool):
        likes_acoustic = acoustic_pref
    else:
        likes_acoustic = float(acoustic_pref) >= 0.5

    return UserProfile(
        favorite_genre=str(user_prefs.get("genre", "")),
        favorite_mood=str(user_prefs.get("mood", "")),
        target_energy=float(user_prefs.get("energy", 0.0)),
        likes_acoustic=likes_acoustic,
    )

def load_songs(csv_path: str) -> List[Dict]:
    """Loads songs from CSV and returns them as typed dictionaries."""
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
                    "lyrics": row["lyrics"],
                    "description": row["description"],
                    "album": row["album"],
                    "release_year": int(row["release_year"]),
                    "popularity": int(row["popularity"]),
                }
            )

    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, str]:
    """Computes a 0-10 match score and explanation for one song."""
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
    """Returns top-k song recommendations as (song, score, explanation) tuples."""
    if k <= 0 or not songs:
        return []

    song_objects = [_song_from_dict(song) for song in songs]
    user_profile = _user_profile_from_prefs(user_prefs)
    recommender = Recommender(song_objects)
    top_song_objects = recommender.recommend(user_profile, k=k)

    results: List[Tuple[Dict, float, str]] = []
    profile_prefs = _prefs_from_user_profile(user_profile)
    for song in top_song_objects:
        song_dict = asdict(song)
        score, _ = score_song(profile_prefs, song_dict)
        explanation = recommender.explain_recommendation(user_profile, song)
        results.append((song_dict, score, explanation))

    return results
