"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv") 
    print(f"Loaded Songs: {len(songs)} ")

    # Taste profile used for content-based comparison
    user_prefs = {
        "genre": "lofi",
        "mood": "focused",
        "energy": 0.41,
        "acousticness": 0.79,
        "danceability": 0.60,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations:\n")
    for i, rec in enumerate(recommendations, 1):
        song, score, explanation = rec
        print(f"{i}. {song['title']}")
        print(f"   Score: {score:.2f}/10")
        print(f"   Why: {explanation}")
        print()


if __name__ == "__main__":
    main()
