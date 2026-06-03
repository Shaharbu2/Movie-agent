import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

# =========================
# 1. Load & prepare data
# =========================

df = pd.read_csv("movies_with_credits_clean.csv")
df["overview"]       = df["overview"].fillna("")
df["genres_clean"]   = df["genres_clean"].fillna("")
df["keywords_clean"] = df["keywords_clean"].fillna("")
df["top_cast"]       = df["top_cast"].fillna("")
df["director"]       = df["director"].fillna("")

# =========================
# 2. Build clusters (same as clustering.py)
# =========================

numeric_features = [
    "runtime", "popularity", "vote_average",
    "vote_count", "budget", "revenue", "genre_count"
]

scaler = MinMaxScaler()
numeric_scaled = pd.DataFrame(
    scaler.fit_transform(df[numeric_features]),
    columns=numeric_features
)

def vectorize_column(col, max_features=None):
    vec = CountVectorizer(
        tokenizer=lambda x: [i.strip() for i in str(x).split(",")],
        token_pattern=None,
        binary=True,
        max_features=max_features
    )
    m = vec.fit_transform(df[col].astype(str))
    return pd.DataFrame(
        m.toarray(),
        columns=[f"{col}_{c}" for c in vec.get_feature_names_out()]
    )

genres_vector   = vectorize_column("genres_clean")
keywords_vector = vectorize_column("keywords_clean", max_features=100)

cluster_data = pd.concat([numeric_scaled, genres_vector, keywords_vector], axis=1)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(cluster_data)

CLUSTER_NAMES = {
    0: "Drama / Crime / History",
    1: "Comedy / Romance",
    2: "Action / Sci-Fi / Thriller",
    3: "Family / Animation / Fantasy",
    4: "Horror / Mystery / Thriller"
}

# =========================
# 3. Build TF-IDF on overviews
# =========================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["overview_clean"] = df["overview"].apply(clean_text)

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)
tfidf_matrix = tfidf.fit_transform(df["overview_clean"])

# =========================
# 4. Genre keyword map
#    Maps words a user might say → actual genre names in the dataset
# =========================

GENRE_KEYWORD_MAP = {
    # Action
    "action"       : "Action",
    "fight"        : "Action",
    "fighting"     : "Action",
    "battle"       : "Action",
    "combat"       : "Action",
    "explosion"    : "Action",
    "chase"        : "Action",
    # Adventure
    "adventure"    : "Adventure",
    "quest"        : "Adventure",
    "journey"      : "Adventure",
    "explore"      : "Adventure",
    "expedition"   : "Adventure",
    # Animation
    "animation"    : "Animation",
    "animated"     : "Animation",
    "cartoon"      : "Animation",
    # Comedy
    "comedy"       : "Comedy",
    "funny"        : "Comedy",
    "humor"        : "Comedy",
    "laugh"        : "Comedy",
    "hilarious"    : "Comedy",
    "fun"          : "Comedy",
    # Crime
    "crime"        : "Crime",
    "criminal"     : "Crime",
    "heist"        : "Crime",
    "robbery"      : "Crime",
    "mafia"        : "Crime",
    "gangster"     : "Crime",
    "detective"    : "Crime",
    # Documentary
    "documentary"  : "Documentary",
    "real story"   : "Documentary",
    "true story"   : "Documentary",
    # Drama
    "drama"        : "Drama",
    "emotional"    : "Drama",
    "serious"      : "Drama",
    "touching"     : "Drama",
    "moving"       : "Drama",
    # Family
    "family"       : "Family",
    "kids"         : "Family",
    "children"     : "Family",
    "child"        : "Family",
    "wholesome"    : "Family",
    # Fantasy
    "fantasy"      : "Fantasy",
    "magic"        : "Fantasy",
    "magical"      : "Fantasy",
    "wizard"       : "Fantasy",
    "dragon"       : "Fantasy",
    "mythical"     : "Fantasy",
    # History
    "history"      : "History",
    "historical"   : "History",
    "war"          : "History",
    "ancient"      : "History",
    "medieval"     : "History",
    # Horror
    "horror"       : "Horror",
    "scary"        : "Horror",
    "terrifying"   : "Horror",
    "ghost"        : "Horror",
    "haunted"      : "Horror",
    "monster"      : "Horror",
    "zombie"       : "Horror",
    # Music
    "music"        : "Music",
    "musical"      : "Music",
    "singer"       : "Music",
    "band"         : "Music",
    # Mystery
    "mystery"      : "Mystery",
    "mysterious"   : "Mystery",
    "suspense"     : "Mystery",
    "puzzle"       : "Mystery",
    "whodunit"     : "Mystery",
    # Romance
    "romance"      : "Romance",
    "romantic"     : "Romance",
    "love story"   : "Romance",
    "love"         : "Romance",
    "relationship" : "Romance",
    # Science Fiction
    "science fiction": "Science Fiction",
    "sci-fi"       : "Science Fiction",
    "scifi"        : "Science Fiction",
    "space"        : "Science Fiction",
    "robot"        : "Science Fiction",
    "alien"        : "Science Fiction",
    "future"       : "Science Fiction",
    "dystopia"     : "Science Fiction",
    "time travel"  : "Science Fiction",
    # Thriller
    "thriller"     : "Thriller",
    "thrilling"    : "Thriller",
    "tense"        : "Thriller",
    "suspenseful"  : "Thriller",
    "spy"          : "Thriller",
    "assassin"     : "Thriller",
    # War
    "war"          : "War",
    "soldier"      : "War",
    "military"     : "War",
    "army"         : "War",
    "wwii"         : "War",
    "world war"    : "War",
    # Western
    "western"      : "Western",
    "cowboy"       : "Western",
    "wild west"    : "Western",
}

# =========================
# 5. Extract genres from user text
# =========================

def extract_genres_from_text(user_text):
    """
    Scan user text for genre-related keywords.
    Returns a list of matched genre names.
    """
    text_lower = user_text.lower()
    matched = set()
    # check multi-word phrases first (longer matches take priority)
    sorted_keys = sorted(GENRE_KEYWORD_MAP.keys(), key=len, reverse=True)
    for keyword in sorted_keys:
        if keyword in text_lower:
            matched.add(GENRE_KEYWORD_MAP[keyword])
    return list(matched)

# =========================
# 6. Map genres → most likely cluster
# =========================

GENRE_TO_CLUSTER = {
    "Action"          : 2,
    "Adventure"       : 2,
    "Science Fiction" : 2,
    "Thriller"        : 2,
    "Comedy"          : 1,
    "Romance"         : 1,
    "Drama"           : 0,
    "Crime"           : 0,
    "History"         : 0,
    "War"             : 0,
    "Family"          : 3,
    "Animation"       : 3,
    "Fantasy"         : 3,
    "Horror"          : 4,
    "Mystery"         : 4,
    "Music"           : 1,
    "Documentary"     : 0,
    "Western"         : 0,
}

def genres_to_clusters(genres):
    """Returns a set of cluster IDs that match the given genres."""
    clusters = set()
    for g in genres:
        if g in GENRE_TO_CLUSTER:
            clusters.add(GENRE_TO_CLUSTER[g])
    return clusters

# =========================
# 7. The smart search function
# =========================

def smart_search(user_text, top_n=5):
    """
    Full pipeline:
    1. Extract genres from user text
    2. Identify matching clusters
    3. Score each movie with a COMBINED score:
         - TF-IDF cosine similarity on overview  (weight: 0.5)
         - Genre match score                     (weight: 0.3)
         - Cluster match bonus                   (weight: 0.2)
    4. Return top_n results
    """

    # --- Step A: genre extraction ---
    matched_genres  = extract_genres_from_text(user_text)
    matched_clusters = genres_to_clusters(matched_genres)

    print(f"\nDetected genres  : {matched_genres if matched_genres else 'none'}")
    print(f"Matching clusters: {[CLUSTER_NAMES[c] for c in matched_clusters] if matched_clusters else 'none'}")

    # --- Step B: TF-IDF score ---
    cleaned = clean_text(user_text)
    user_vec = tfidf.transform([cleaned])
    tfidf_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()

    # --- Step C: genre match score per movie ---
    def genre_match_score(movie_genres_str):
        if not matched_genres:
            return 0.0
        movie_genres = [g.strip() for g in movie_genres_str.split(",")]
        hits = sum(1 for g in matched_genres if g in movie_genres)
        return hits / len(matched_genres)

    genre_scores = df["genres_clean"].apply(genre_match_score).values

    # --- Step D: cluster match score per movie ---
    def cluster_match_score(cluster_id):
        if not matched_clusters:
            return 0.0
        return 1.0 if cluster_id in matched_clusters else 0.0

    cluster_scores = df["cluster"].apply(cluster_match_score).values

    # --- Step E: combine with weights ---
    combined = (
        0.5 * tfidf_scores +
        0.3 * genre_scores  +
        0.2 * cluster_scores
    )

    top_indices = combined.argsort()[-top_n:][::-1]

    results = df.iloc[top_indices][[
        "title", "release_year", "genres_clean",
        "director", "vote_average", "overview"
    ]].copy()

    results["tfidf_score"]   = tfidf_scores[top_indices].round(3)
    results["genre_score"]   = genre_scores[top_indices].round(3)
    results["cluster_score"] = cluster_scores[top_indices].round(3)
    results["combined_score"]= combined[top_indices].round(3)
    results["overview"]      = results["overview"].str[:120] + "..."

    return results, matched_genres, matched_clusters

# =========================
# 8. Demo
# =========================

test_queries = [
    "I want a funny romantic movie with a love story",
    "scary ghost movie with mystery and suspense",
    "space adventure with robots and aliens fighting",
    "animated film for kids with magic and dragons",
    "a true war story about soldiers in world war"
]

print("=" * 65)
print("SMART SEARCH DEMO")
print("=" * 65)

for query in test_queries:
    print(f"\n{'='*65}")
    print(f"Query: \"{query}\"")
    results, genres, clusters = smart_search(query, top_n=5)
    print(results[["title","release_year","genres_clean",
                    "vote_average","combined_score"]].to_string(index=False))

print("\nSmart search module ready.")
