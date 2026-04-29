"""
Embedding Pipeline: Convert songs to embeddings and store in vector database.

This module handles:
1. Converting song metadata from csv into text documents
2. Generating embeddings using SentenceTransformer
3. Storing embeddings in FAISS vector database
4. Saving/loading the vector DB for reuse(will use in RAG retrieval)

Note: Requires sentence-transformers and faiss-cpu packages.
Install with: pip install sentence-transformers faiss-cpu
"""

from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
import pickle
import json
from dataclasses import asdict

# These will need to be added to requirements.txt
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    raise ImportError(
        "Please install required packages:\n"
        "pip install sentence-transformers faiss-cpu"
    )


def song_to_text(song: Dict) -> str:
    """
    Convert a song dictionary into a text representation for embedding.
    
    Combines multiple song attributes into a single text string that captures
    the song's semantic meaning. This text will be encoded into a 384-dimensional
    embedding vector.
    
    Strategy: Weight high-value fields (title, artist) with repetition, and include
    categorical features (genre, mood) to improve semantic retrieval.
    
    Args:
        song (Dict): Song dictionary with keys: title, artist, genre, mood, description, album
        
    Returns:
        str: Combined text representation of the song
        
    Example:
        song = {"title": "Blinding Lights", "artist": "The Weekend", "genre": "pop", "mood": "happy", ...}
        text = song_to_text(song)
        # Returns: "Blinding Lights The Weeknd pop happy energetic The Weeknd Blinding Lights..."
    """
    title = song.get("title", "").strip()
    artist = song.get("artist", "").strip()
    genre = song.get("genre", "").strip()
    mood = song.get("mood", "").strip()
    description = song.get("description", "").strip()
    album = song.get("album", "").strip()
    
    # Build text with weighted importance: title and artist repeated for emphasis
    text_parts = [
        title,           # Song title (primary)
        artist,          # Artist name (primary)
        genre,           # Genre (categorical)
        mood,            # Mood (categorical)
        title,           # Repeat title for emphasis
        artist,          # Repeat artist for emphasis
        description,     # Album/song description
        album,           # Album name
    ]
    
    # Filter out empty strings and join
    text = " ".join([part for part in text_parts if part])
    return text


def load_and_embed_songs(
    csv_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Load songs from CSV and convert them into embeddings.
    
    Process:
    1. Load all songs from CSV file
    2. Convert each song to text representation
    3. Use SentenceTransformer to encode all texts into 384-dimensional vectors
    4. Return embeddings and original song data
    
    The SentenceTransformer model 'all-MiniLM-L6-v2' produces 384-dimensional
    embeddings optimized for semantic similarity (fast and accurate for this use case).
    
    Args:
        csv_path (str): Path to the songs CSV file (e.g., "data/songs.csv")
        model_name (str): SentenceTransformer model to use (default: all-MiniLM-L6-v2)
        batch_size (int): Number of songs to process at once (default: 32)
        
    Returns:
        Tuple[np.ndarray, List[Dict]]: 
            - embeddings: Shape (num_songs, 384) float32 array
            - songs: List of song dictionaries (in same order as embeddings)
            
    Example:
        embeddings, songs = load_and_embed_songs("data/songs.csv")
        print(f"Created {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    """
    from recommender import load_songs
    
    print(f"Loading songs from {csv_path}...")
    songs = load_songs(csv_path)
    print(f"Loaded {len(songs)} songs")
    
    print(f"Initializing SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("Converting songs to text representations...")
    song_texts = [song_to_text(song) for song in songs]
    
    print(f"Encoding {len(song_texts)} songs into embeddings (batch_size={batch_size})...")
    embeddings = model.encode(song_texts, batch_size=batch_size, show_progress_bar=True)
    
    # Convert to float32 for FAISS compatibility
    embeddings = embeddings.astype(np.float32)
    
    print(f"✓ Created embeddings with shape: {embeddings.shape}")
    print(f"  - {embeddings.shape[0]} songs")
    print(f"  - {embeddings.shape[1]} dimensions per embedding")
    
    return embeddings, songs


def create_vector_db(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Create a FAISS vector database index from embeddings.
    
    FAISS (Facebook AI Similarity Search) stores embeddings and allows fast
    nearest-neighbor retrieval using L2 (Euclidean) distance.
    
    IndexFlatL2: Simple but accurate - compares against all stored vectors
    (good for ~500 songs; can scale to millions if needed).
    
    Args:
        embeddings (np.ndarray): Shape (num_songs, 384) array of embeddings
        
    Returns:
        faiss.IndexFlatL2: FAISS index ready for querying
        
    Example:
        index = create_vector_db(embeddings)
        distances, indices = index.search(query_vector, k=10)
    """
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    
    embedding_dim = embeddings.shape[1]
    print(f"Creating FAISS IndexFlatL2 with dimension {embedding_dim}...")
    
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    
    print(f"✓ FAISS index created and populated with {index.ntotal} embeddings")
    return index


def save_vector_db(
    index: faiss.IndexFlatL2,
    songs: List[Dict],
    index_path: str,
    metadata_path: str
) -> None:
    """
    Save FAISS vector database and song metadata to disk.
    
    Saves two files:
    1. index_path: FAISS index (binary format, ~500KB for 500 songs)
    2. metadata_path: Song data as JSON (for retrieving original song info during retrieval)
    
    This allows you to load the pre-computed embeddings without re-encoding all songs.
    
    Args:
        index (faiss.IndexFlatL2): FAISS index to save
        songs (List[Dict]): List of song dictionaries
        index_path (str): Where to save the FAISS index (e.g., "data/songs_index.faiss")
        metadata_path (str): Where to save song metadata (e.g., "data/songs_metadata.json")
        
    Example:
        save_vector_db(index, songs, "data/songs_index.faiss", "data/songs_metadata.json")
    """
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)
    
    print(f"Saving song metadata to {metadata_path}...")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(songs, f, indent=2)
    
    print(f"✓ Saved index ({Path(index_path).stat().st_size / 1024:.1f} KB)")
    print(f"✓ Saved metadata ({Path(metadata_path).stat().st_size / 1024:.1f} KB)")


def load_vector_db(index_path: str, metadata_path: str) -> Tuple[faiss.IndexFlatL2, List[Dict]]:
    """
    Load pre-computed FAISS vector database and song metadata from disk.
    
    This is much faster than re-embedding all songs. Call this at startup
    to load the knowledge base once.
    
    Args:
        index_path (str): Path to saved FAISS index (e.g., "data/songs_index.faiss")
        metadata_path (str): Path to saved metadata (e.g., "data/songs_metadata.json")
        
    Returns:
        Tuple[faiss.IndexFlatL2, List[Dict]]:
            - index: FAISS index ready for querying
            - songs: List of song dictionaries (in same order as index)
            
    Example:
        index, songs = load_vector_db("data/songs_index.faiss", "data/songs_metadata.json")
        print(f"Loaded {index.ntotal} songs from vector DB")
    """
    print(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(index_path)
    
    print(f"Loading song metadata from {metadata_path}...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        songs = json.load(f)
    
    print(f"✓ Loaded {index.ntotal} songs from vector database")
    return index, songs


def query_vector_db(
    query_embedding: np.ndarray,
    index: faiss.IndexFlatL2,
    songs: List[Dict],
    k: int = 10
) -> List[Tuple[Dict, float]]:
    """
    Query the vector database to find the k most similar songs.
    
    Uses L2 (Euclidean) distance: smaller distances = more similar songs.
    Returns both the song data and similarity distance.
    
    Args:
        query_embedding (np.ndarray): Shape (1, 384) embedding of the query
        index (faiss.IndexFlatL2): FAISS index to search
        songs (List[Dict]): List of all songs (in same order as index)
        k (int): Number of results to return (default: 10)
        
    Returns:
        List[Tuple[Dict, float]]: List of (song, distance) tuples, sorted by distance (closest first)
        
    
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    query_embedding = query_embedding.astype(np.float32)
    
    distances, indices = index.search(query_embedding, k)
    
    # Flatten results (FAISS returns 2D arrays)
    distances = distances[0]
    indices = indices[0]
    
    # Return list of (song, distance) tuples
    results = [
        (songs[idx], float(dist))
        for idx, dist in zip(indices, distances)
    ]
    
    return results


def build_knowledge_base(
    csv_path: str = "data/songs.csv",
    index_path: str = "data/songs_index.faiss",
    metadata_path: str = "data/songs_metadata.json"
) -> Tuple[faiss.IndexFlatL2, List[Dict]]:
    """
    Build the complete knowledge base: embed all songs and create vector DB.
    
    One-time setup process:
    1. Load songs from CSV
    2. Convert to text and create embeddings
    3. Create FAISS index
    4. Save to disk
    
    On subsequent runs, just call load_vector_db() to reuse these embeddings.
    
    Args:
        csv_path (str): Path to songs CSV
        index_path (str): Where to save FAISS index
        metadata_path (str): Where to save song metadata
        
    Returns:
        Tuple[faiss.IndexFlatL2, List[Dict]]: Index and songs (already saved to disk)
        
    Example:
        # First time: build and save
        index, songs = build_knowledge_base()
        
        # Later runs: just load
        index, songs = load_vector_db("data/songs_index.faiss", "data/songs_metadata.json")
    """
    print("\n" + "="*60)
    print("BUILDING KNOWLEDGE BASE: Converting songs to embeddings")
    print("="*60 + "\n")
    
    embeddings, songs = load_and_embed_songs(csv_path)
    index = create_vector_db(embeddings)
    save_vector_db(index, songs, index_path, metadata_path)
    
    print("\n" + "="*60)
    print("✓ KNOWLEDGE BASE COMPLETE")
    print("="*60 + "\n")
    
    return index, songs


if __name__ == "__main__":
    # Example usage: build knowledge base from scratch
    index, songs = build_knowledge_base()
    
    # Example query
    print("\nExample Query:")
    print("-" * 60)
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    query_text = "upbeat dance music for the gym"
    query_emb = model.encode(query_text)
    
    results = query_vector_db(query_emb, index, songs, k=5)
    
    print(f"Query: '{query_text}'")
    print(f"\nTop 5 matches:")
    for i, (song, distance) in enumerate(results, 1):
        print(f"  {i}. {song['title']} - {song['artist']}")
        print(f"     Genre: {song['genre']}, Mood: {song['mood']}")
        print(f"     Distance (L2): {distance:.4f}")
        print()
