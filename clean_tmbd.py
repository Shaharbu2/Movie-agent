import pandas as pd
import ast


# =========================
# 1. Load files
# =========================

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")


# =========================
# 2. Helper functions
# =========================

def extract_names(json_text):
    try:
        items = ast.literal_eval(json_text)
        return ", ".join([item["name"] for item in items])
    except:
        return ""


def extract_top_cast(json_text, top_n=5):
    try:
        items = ast.literal_eval(json_text)
        return ", ".join([item["name"] for item in items[:top_n]])
    except:
        return ""


def extract_director(json_text):
    try:
        items = ast.literal_eval(json_text)
        for item in items:
            if item.get("job") == "Director":
                return item.get("name")
        return ""
    except:
        return ""


# =========================
# 3. Clean movies file
# =========================

movies_cols = [
    "id",
    "title",
    "budget",
    "genres",
    "keywords",
    "original_language",
    "overview",
    "popularity",
    "release_date",
    "revenue",
    "runtime",
    "status",
    "vote_average",
    "vote_count"
]

movies_clean = movies[movies_cols].copy()

movies_clean = movies_clean.drop_duplicates()
movies_clean = movies_clean.dropna(subset=["id", "title"])

movies_clean["release_date"] = pd.to_datetime(
    movies_clean["release_date"],
    errors="coerce"
)

movies_clean["release_year"] = movies_clean["release_date"].dt.year

numeric_cols = [
    "budget",
    "popularity",
    "revenue",
    "runtime",
    "vote_average",
    "vote_count"
]

for col in numeric_cols:
    movies_clean[col] = pd.to_numeric(movies_clean[col], errors="coerce")

movies_clean = movies_clean.dropna(subset=[
    "runtime",
    "vote_average",
    "vote_count",
    "popularity",
    "release_year"
])

movies_clean = movies_clean[
    (movies_clean["runtime"] > 0) &
    (movies_clean["vote_average"].between(0, 10)) &
    (movies_clean["vote_count"] > 0) &
    (movies_clean["release_year"] >= 1900)
]

movies_clean["genres_clean"] = movies_clean["genres"].apply(extract_names)
movies_clean["keywords_clean"] = movies_clean["keywords"].apply(extract_names)

movies_clean["overview"] = movies_clean["overview"].fillna("")
movies_clean["original_language"] = movies_clean["original_language"].fillna("unknown")
movies_clean["status"] = movies_clean["status"].fillna("unknown")

movies_clean["genre_count"] = movies_clean["genres_clean"].apply(
    lambda x: len(x.split(", ")) if x != "" else 0
)

movies_clean["quality_category"] = pd.cut(
    movies_clean["vote_average"],
    bins=[0, 5, 7, 10],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)


# =========================
# 4. Clean credits file
# =========================

credits_clean = credits.copy()

credits_clean = credits_clean.drop_duplicates()
credits_clean = credits_clean.dropna(subset=["movie_id", "title"])

credits_clean["top_cast"] = credits_clean["cast"].apply(
    lambda x: extract_top_cast(x, top_n=5)
)

credits_clean["director"] = credits_clean["crew"].apply(extract_director)

credits_clean = credits_clean[
    ["movie_id", "title", "top_cast", "director"]
]


# =========================
# 5. Merge movies + credits
# =========================

final_df = movies_clean.merge(
    credits_clean,
    left_on="id",
    right_on="movie_id",
    how="inner",
    suffixes=("", "_credits")
)

final_df = final_df.drop(columns=["title_credits", "movie_id"])

final_df = final_df.dropna(subset=["director"])

final_df["director"] = final_df["director"].fillna("")
final_df["top_cast"] = final_df["top_cast"].fillna("")

final_df = final_df[
    final_df["director"] != ""
]


# =========================
# 6. Select final columns
# =========================

final_cols = [
    "id",
    "title",
    "release_year",
    "runtime",
    "original_language",
    "genres_clean",
    "keywords_clean",
    "overview",
    "top_cast",
    "director",
    "popularity",
    "vote_average",
    "vote_count",
    "budget",
    "revenue",
    "genre_count",
    "quality_category"
]

final_df = final_df[final_cols].copy()


# =========================
# 7. Save clean file
# =========================

final_df.to_csv(
    "movies_with_credits_clean.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================
# 8. Descriptive checks
# =========================

print("Cleaning completed successfully.")
print("Rows:", len(final_df))
print("Columns:", len(final_df.columns))

print("\nMissing values:")
print(final_df.isna().sum())

print("\nTop 10 directors:")
print(final_df["director"].value_counts().head(10))

print("\nVote average statistics:")
print(final_df["vote_average"].describe())

print("\nRuntime statistics:")
print(final_df["runtime"].describe())

print("\nFirst rows:")
print(final_df.head())

