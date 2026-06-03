import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1. Load clean data
# =========================

df = pd.read_csv("movies_with_credits_clean.csv")


# =========================
# 2. Choose features
# =========================

numeric_features = [
    "runtime",
    "popularity",
    "vote_average",
    "vote_count",
    "budget",
    "revenue",
    "genre_count"
]

category_feature = "genres_clean"

text_features = [
    "keywords_clean",
    "top_cast",
    "director",
    "quality_category"
]


# =========================
# 3. Normalize numeric columns
# =========================

scaler = MinMaxScaler()

numeric_scaled = pd.DataFrame(
    scaler.fit_transform(df[numeric_features]),
    columns=numeric_features
)


# =========================
# 4. Convert category/text columns to numeric vectors
# =========================

def vectorize_column(column_name, max_features=None):
    vectorizer = CountVectorizer(
        tokenizer=lambda x: [item.strip() for item in str(x).split(",")],
        token_pattern=None,
        binary=True,
        max_features=max_features
    )

    matrix = vectorizer.fit_transform(df[column_name].astype(str))

    return pd.DataFrame(
        matrix.toarray(),
        columns=[f"{column_name}_{c}" for c in vectorizer.get_feature_names_out()]
    )


genres_vector = vectorize_column("genres_clean")
keywords_vector = vectorize_column("keywords_clean", max_features=300)
cast_vector = vectorize_column("top_cast", max_features=300)
director_vector = vectorize_column("director", max_features=300)
quality_vector = vectorize_column("quality_category")


# =========================
# 5. Scenario A - WITH category
# =========================

data_with_category = pd.concat(
    [
        numeric_scaled,
        genres_vector,
        keywords_vector,
        cast_vector,
        director_vector,
        quality_vector
    ],
    axis=1
)


# =========================
# 6. Scenario B - WITHOUT category
# =========================

data_without_category = pd.concat(
    [
        numeric_scaled,
        keywords_vector,
        cast_vector,
        director_vector,
        quality_vector
    ],
    axis=1
)


# =========================
# 7. Cosine Similarity
# =========================

def get_similar_cosine(movie_title, data, top_n=5):
    matches = df[df["title"].str.lower().str.contains(movie_title.lower(), na=False)]

    if matches.empty:
        print("Movie not found")
        return None

    idx = matches.index[0]

    scores = cosine_similarity(
        data.iloc[idx:idx + 1],
        data
    ).flatten()

    scores[idx] = -1

    similar_indices = scores.argsort()[-top_n:][::-1]

    results = df.iloc[similar_indices][[
        "title",
        "release_year",
        "genres_clean",
        "director",
        "top_cast",
        "vote_average",
        "popularity"
    ]].copy()

    results["similarity_score"] = scores[similar_indices]

    return results


# =========================
# 8. Jaccard Similarity
# =========================

def get_similar_jaccard(movie_title, top_n=5):
    matches = df[df["title"].str.lower().str.contains(movie_title.lower(), na=False)]

    if matches.empty:
        print("Movie not found")
        return None

    idx = matches.index[0]

    target = genres_vector.iloc[idx].values.astype(bool)

    scores = []

    for i in range(len(df)):
        if i == idx:
            scores.append(-1)
            continue

        current = genres_vector.iloc[i].values.astype(bool)

        intersection = np.logical_and(target, current).sum()
        union = np.logical_or(target, current).sum()

        if union == 0:
            score = 0
        else:
            score = intersection / union

        scores.append(score)

    scores = np.array(scores)

    similar_indices = scores.argsort()[-top_n:][::-1]

    results = df.iloc[similar_indices][[
        "title",
        "release_year",
        "genres_clean",
        "director",
        "top_cast",
        "vote_average",
        "popularity"
    ]].copy()

    results["similarity_score"] = scores[similar_indices]

    return results


# =========================
# 9. Run example
# =========================

movie_to_test = "Avatar"

print("\n==============================")
print("Movie checked:", movie_to_test)
print("==============================")

print("\n--- Movie details ---")
print(
    df[df["title"].str.contains(movie_to_test, case=False, na=False)][[
        "title",
        "release_year",
        "genres_clean",
        "keywords_clean",
        "director",
        "top_cast",
        "vote_average"
    ]].head(1)
)

print("\n--- Cosine Similarity WITH category / genres ---")
print(get_similar_cosine(movie_to_test, data_with_category))

print("\n--- Cosine Similarity WITHOUT category / genres ---")
print(get_similar_cosine(movie_to_test, data_without_category))

print("\n--- Jaccard Similarity based on genres only ---")
print(get_similar_jaccard(movie_to_test))

print("\nSimilarity analysis completed successfully.")

data_with_category.to_csv(
    "movies_similarity_with_category.csv",
    index=False
)

data_without_category.to_csv(
    "movies_similarity_without_category.csv",
    index=False
)

print("Similarity datasets saved.")