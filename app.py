"""
Music Recommender — Streamlit Frontend

Two-step flow:
  Step 1: User sets music preferences (genre, mood, energy, acousticness, danceability).
          These are stored in st.session_state and used to re-rank results in Phase 5.
  Step 2: User enters a natural language query or pastes a Spotify playlist URL.
          The full pipeline runs: parse → embed → retrieve → rerank → display.
"""

import sys
import os

# Make bare-name imports work the same way the src modules import each other
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from dotenv import load_dotenv

from embedding_pipeline import load_vector_db
from query_embedder import load_embedding_model, build_query_embedding
from query_processor import parse_user_input, fetch_playlist_tracks, match_tracks_to_csv
from rag_retriever import retrieve_candidates
from reranker import rerank_candidates

load_dotenv()

# Paths to the pre-built FAISS index (created by build_knowledge_base() in embedding_pipeline.py)
_INDEX_PATH = "data/songs_index.faiss"
_METADATA_PATH = "data/songs_metadata.json"


# ---------------------------------------------------------------------------
# Cached resource loading — runs once per Streamlit session, not per rerun
# ---------------------------------------------------------------------------

@st.cache_resource
def load_resources():
    """Load FAISS index, song metadata, and the SentenceTransformer model.

    Returns (None, None, None) if the index files haven't been built yet so the
    caller can show a helpful error instead of crashing.
    """
    if not os.path.exists(_INDEX_PATH) or not os.path.exists(_METADATA_PATH):
        return None, None, None
    index, songs = load_vector_db(_INDEX_PATH, _METADATA_PATH)
    model = load_embedding_model()
    return index, songs, model


@st.cache_data
def get_genres_and_moods(_songs):
    """Extract sorted unique genre and mood values from the loaded song catalogue.

    The leading underscore on _songs tells Streamlit not to try to hash the list
    (unhashable), so changes to songs don't bust the cache unexpectedly.
    """
    genres = sorted({s["genre"] for s in _songs if s.get("genre")})
    moods  = sorted({s["mood"]  for s in _songs if s.get("mood")})
    return genres, moods


# ---------------------------------------------------------------------------
# Step 1: Collect user preferences
# ---------------------------------------------------------------------------

def show_preferences_form(genres: list, moods: list) -> None:
    """Render the preference form.  On submit, saves prefs to session_state and reruns."""
    st.title("Music Recommender")
    st.subheader("Step 1 of 2 — Set your music taste")
    st.write(
        "Your answers below tell the recommender which songs to rank highest. "
        "You can change them at any time from the sidebar."
    )

    with st.form("preferences_form"):
        genre = st.selectbox("Favorite Genre", genres)
        mood  = st.selectbox("Favorite Mood",  moods)

        # Sliders — descriptive labels explain the 0→1 scale at a glance
        energy       = st.slider("Energy  (calm → energetic)",       0.0, 1.0, 0.5, step=0.05)
        acousticness = st.slider("Acousticness  (electronic → acoustic)", 0.0, 1.0, 0.5, step=0.05)
        danceability = st.slider("Danceability  (chill → very danceable)", 0.0, 1.0, 0.5, step=0.05)

        submitted = st.form_submit_button("Save Preferences →", type="primary")

    if submitted:
        # Store as a plain dict — this is the user_prefs shape score_song() expects
        st.session_state.prefs = {
            "genre":        genre,
            "mood":         mood,
            "energy":       energy,
            "acousticness": acousticness,
            "danceability": danceability,
        }
        st.rerun()  # jump to step 2


# ---------------------------------------------------------------------------
# Step 2: Query input and results
# ---------------------------------------------------------------------------

def show_query_interface(index, songs: list, model) -> None:
    """Render the query box and handle the recommendation pipeline."""
    prefs = st.session_state.prefs

    # Sidebar: always-visible preferences summary + reset button
    with st.sidebar:
        st.header("Your Preferences")
        st.write(f"**Genre:** {prefs['genre'].title()}")
        st.write(f"**Mood:** {prefs['mood'].title()}")
        st.write(f"**Energy:** {prefs['energy']:.2f}")
        st.write(f"**Acousticness:** {prefs['acousticness']:.2f}")
        st.write(f"**Danceability:** {prefs['danceability']:.2f}")
        st.divider()
        if st.button("Change Preferences"):
            del st.session_state.prefs
            st.rerun()

    # Main area
    st.title("Music Recommender")
    st.subheader("Step 2 of 2 — What are you looking for?")

    query = st.text_input(
        "Describe what you want, or paste a Spotify playlist URL",
        placeholder="e.g. 'chill songs for late-night studying'  or  https://open.spotify.com/playlist/...",
    )

    if st.button("Get Recommendations", type="primary"):
        if not query.strip():
            st.warning("Please enter a query or playlist URL before searching.")
            return
        _run_pipeline(query.strip(), prefs, index, songs, model)


def _run_pipeline(query: str, user_prefs: dict, index, songs: list, model) -> None:
    """Execute Phases 2–5 and render the ranked results."""

    with st.spinner("Finding your music..."):

        # Phase 2: detect input format and split URL from context text
        parsed = parse_user_input(query)

        matched_csv = []
        unmatched   = []

        # Playlist URL path: fetch Spotify tracks, match against the local CSV
        if parsed["input_type"] == "playlist_url":
            try:
                playlist_tracks = fetch_playlist_tracks(parsed["playlist_id"])
                matched_csv, unmatched = match_tracks_to_csv(playlist_tracks, songs)
            except EnvironmentError:
                st.error(
                    "Spotify credentials not found. "
                    "Add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to your .env file."
                )
                return
            except Exception as exc:
                st.error(f"Could not fetch playlist from Spotify: {exc}")
                return

        # Phase 3: convert the query into a 384-dim embedding
        query_embedding = build_query_embedding(parsed, matched_csv, unmatched, model)

        # Phase 4: semantic retrieval — top 20 candidates by L2 distance
        candidates = retrieve_candidates(query_embedding, index, songs, k=20)

        # Phase 5: re-rank the 20 candidates by user preference score (0–10)
        results = rerank_candidates(candidates, user_prefs, k=10)

    # Display ranked results
    st.subheader(f"Top {len(results)} Recommendations")

    for i, (song, score, explanation) in enumerate(results, 1):
        with st.container():
            col_info, col_score = st.columns([3, 1])

            with col_info:
                st.markdown(f"**{i}. {song['title']}** — *{song['artist']}*")
                st.caption(
                    f"{song['genre'].title()}  ·  {song['mood'].title()}  ·  "
                    f"{song.get('album', '')}  ·  {song.get('release_year', '')}"
                )
                # Explanation from score_song() shows which attributes matched
                st.caption(f"Why: {explanation}")

            with col_score:
                st.metric("Score", f"{score:.1f} / 10")

            st.divider()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Music Recommender",
        page_icon="🎵",
        layout="centered",
    )

    # Load the knowledge base once (cached for the session lifetime)
    index, songs, model = load_resources()

    # Guard: FAISS index must be built before the app can function
    if index is None:
        st.title("Music Recommender")
        st.error("Knowledge base not found — the FAISS index has not been built yet.")
        st.info(
            "Run this once from the project root to build the index:\n\n"
            "```python\n"
            "from src.embedding_pipeline import build_knowledge_base\n"
            "build_knowledge_base()\n"
            "```"
        )
        return

    genres, moods = get_genres_and_moods(songs)

    # Route to step 1 (preferences) or step 2 (query) based on session state
    if "prefs" not in st.session_state:
        show_preferences_form(genres, moods)
    else:
        show_query_interface(index, songs, model)


if __name__ == "__main__":
    main()
