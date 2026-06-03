import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

# =========================
# 1. Load data
# =========================

df = pd.read_csv("movies_with_credits_clean.csv")

for col in ["genres_clean", "keywords_clean"]:
    df[col] = df[col].fillna("")

# =========================
# 2. Prepare features
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

genres_vector = vectorize_column("genres_clean")
keywords_vector = vectorize_column("keywords_clean", max_features=100)

data = pd.concat([numeric_scaled, genres_vector, keywords_vector], axis=1)

# =========================
# 3. K-Means with K=5
# =========================

K = 5
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(data)

# =========================
# 4. Describe each cluster
# =========================

print("=" * 50)
print("CLUSTER SUMMARY")
print("=" * 50)

for c in range(K):
    cluster_df = df[df["cluster"] == c]
    print(f"\n--- Cluster {c} ({len(cluster_df)} movies) ---")
    print("Top genres:")
    all_genres = ", ".join(cluster_df["genres_clean"].dropna()).split(", ")
    from collections import Counter
    top_genres = Counter(all_genres).most_common(5)
    for g, count in top_genres:
        if g:
            print(f"  {g}: {count}")
    print(f"Avg vote_average : {cluster_df['vote_average'].mean():.2f}")
    print(f"Avg popularity   : {cluster_df['popularity'].mean():.2f}")
    print(f"Avg budget       : ${cluster_df['budget'].mean():,.0f}")
    print(f"Avg runtime      : {cluster_df['runtime'].mean():.0f} min")
    print("Sample movies:")
    print(cluster_df[["title", "release_year", "genres_clean", "vote_average"]]
          .sort_values("vote_average", ascending=False).head(5).to_string(index=False))

# =========================
# 5. Save result
# =========================

df[["id", "title", "release_year", "genres_clean", "vote_average",
    "popularity", "budget", "cluster"]].to_csv(
    "movies_with_clusters.csv", index=False, encoding="utf-8-sig"
)

print("\nClustering completed. File saved: movies_with_clusters.csv")
