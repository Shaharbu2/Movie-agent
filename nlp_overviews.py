import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# =========================
# 1. Load data (with clusters)
# =========================

df = pd.read_csv("movies_with_credits_clean.csv")
clusters_df = pd.read_csv("movies_with_clusters.csv")[["id", "cluster"]]
df = df.merge(clusters_df, on="id", how="left")
df["overview"] = df["overview"].fillna("")

# =========================
# 2. Clean overview text
# =========================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["overview_clean"] = df["overview"].apply(clean_text)

# =========================
# 3. TF-IDF on overviews
# =========================

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)   # single words AND two-word phrases
)

tfidf_matrix = tfidf.fit_transform(df["overview_clean"])
feature_names = tfidf.get_feature_names_out()

print("TF-IDF matrix shape:", tfidf_matrix.shape)

# =========================
# 4. Top keywords per cluster
# =========================

print("\n" + "=" * 50)
print("TOP KEYWORDS PER CLUSTER (from overviews)")
print("=" * 50)

cluster_names = {
    0: "Drama / Crime / History",
    1: "Comedy / Romance",
    2: "Action / Sci-Fi / Thriller",
    3: "Family / Animation / Fantasy",
    4: "Horror / Mystery / Thriller"
}

for c in sorted(df["cluster"].dropna().unique()):
    c = int(c)
    cluster_indices = df[df["cluster"] == c].index
    cluster_matrix = tfidf_matrix[cluster_indices]
    mean_tfidf = cluster_matrix.mean(axis=0).A1
    top_indices = mean_tfidf.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    print(f"\nCluster {c} – {cluster_names.get(c, '')} ({len(cluster_indices)} movies)")
    print("  Keywords:", ", ".join(top_words))

# =========================
# 5. Text-based movie finder
# =========================

def find_movies_by_description(user_text, top_n=5):
    """
    Given a free-text description from the user,
    find the most similar movies by overview TF-IDF.
    """
    cleaned = clean_text(user_text)
    user_vec = tfidf.transform([cleaned])
    scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[-top_n:][::-1]

    results = df.iloc[top_indices][
        ["title", "release_year", "genres_clean", "director", "vote_average"]
    ].copy()
    results["text_similarity"] = scores[top_indices]
    return results

# =========================
# 6. Test the text finder
# =========================

print("\n" + "=" * 50)
print("NLP TEXT SEARCH DEMO")
print("=" * 50)

queries = [
    "a young boy discovers he has magical powers and goes to a special school",
    "a group of criminals plan a bank heist but things go wrong",
    "robots and humans fight in a post-apocalyptic world"
]

for q in queries:
    print(f"\nQuery: \"{q}\"")
    print(find_movies_by_description(q).to_string(index=False))

# =========================
# 7. Save TF-IDF scores (top 20 features per movie)
# =========================

print("\nNLP analysis completed successfully.")
