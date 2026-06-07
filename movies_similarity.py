import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix


# =========================
# Load final dataset
# =========================

df = pd.read_csv("movies_master.csv")

df = df[
    [
        "title",
        "overview",
        "genres",
        "keywords",
        "release_year",
        "vote_average",
        "popularity",
        "runtime",
        "available_on"
    ]
].copy()


# =========================
# Clean missing values
# =========================

df["overview"] = df["overview"].fillna("")
df["genres"] = df["genres"].fillna("")
df["keywords"] = df["keywords"].fillna("")
df["available_on"] = df["available_on"].fillna("Not available in dataset")

df["vote_average"] = df["vote_average"].fillna(0)
df["popularity"] = df["popularity"].fillna(0)
df["runtime"] = df["runtime"].fillna(0)


# =========================
# Prepare features
# =========================

df["combined_features"] = (
    df["genres"].astype(str) + " " +
    df["keywords"].astype(str) + " " +
    df["overview"].astype(str)
)


# Scale numeric features
numeric_features = ["vote_average", "popularity", "runtime"]

scaler = MinMaxScaler()
numeric_scaled = scaler.fit_transform(df[numeric_features])
numeric_sparse = csr_matrix(numeric_scaled)


# Convert text into numeric vectors
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

tfidf_matrix = vectorizer.fit_transform(df["combined_features"])


# Combine text and numeric features
final_features = hstack([
    tfidf_matrix,
    numeric_sparse
])


# =========================
# Detect genre from user input
# =========================

def detect_genre(user_input):
    user_input = user_input.lower()

    genre_keywords = {
        "action": "Action",
        "romantic": "Romance",
        "romance": "Romance",
        "love": "Romance",
        "comedy": "Comedy",
        "funny": "Comedy",
        "drama": "Drama",
        "horror": "Horror",
        "scary": "Horror",
        "thriller": "Thriller",
        "sci fi": "Science Fiction",
        "science fiction": "Science Fiction",
        "fantasy": "Fantasy",
        "animation": "Animation",
        "family": "Family",
        "crime": "Crime",
        "mystery": "Mystery",
        "adventure": "Adventure"
    }

    for word, genre in genre_keywords.items():
        if word in user_input:
            return genre

    return None


# =========================
# Recommend movies
# =========================

def recommend_movies(reference_movie, requested_genre=None, top_n=3):
    reference_movie = reference_movie.lower().strip()

    matches = df[
        df["title"].str.lower().str.contains(reference_movie, na=False)
    ]

    if matches.empty:
        return "I could not find this movie in the dataset. Please try another movie name."

    movie_index = matches.index[0]

    similarity_scores = cosine_similarity(
        final_features[movie_index],
        final_features
    ).flatten()

    candidates = df.copy()
    candidates["similarity_score"] = similarity_scores

    # Remove the selected movie itself
    candidates = candidates[candidates.index != movie_index]

    # Filter by requested genre
    if requested_genre is not None:
        candidates = candidates[
            candidates["genres"].str.contains(
                requested_genre,
                case=False,
                na=False
            )
        ]

    candidates = candidates.sort_values(
        by="similarity_score",
        ascending=False
    )

    top_results = candidates.head(top_n)

    recommendations = []

    for _, movie in top_results.iterrows():
        recommendations.append(
            {
                "title": movie["title"],
                "year": int(movie["release_year"]),
                "genres": movie["genres"],
                "rating": round(movie["vote_average"], 1),
                "available_on": movie["available_on"],
                "similarity_score": round(movie["similarity_score"], 3)
            }
        )

    return recommendations


# =========================
# Movie agent
# =========================

def movie_agent(user_input):
    user_input_lower = user_input.lower()

    found_movie = None

    # Find movie title inside user input
    for title in df["title"].dropna().unique():
        if str(title).lower() in user_input_lower:
            found_movie = title
            break

    if found_movie is None:
        return (
            "I could not identify a movie you liked from your message.\n"
            "Please write a movie name, for example: I liked Me Before You."
        )

    requested_genre = detect_genre(user_input)

    recommendations = recommend_movies(
        found_movie,
        requested_genre=requested_genre,
        top_n=3
    )

    if isinstance(recommendations, str):
        return recommendations

    answer = f"Based on the movie '{found_movie}'"

    if requested_genre is not None:
        answer += f" and your request for a {requested_genre} movie"

    answer += ", I recommend:\n\n"

    for i, movie in enumerate(recommendations, start=1):
        answer += (
            f"{i}. {movie['title']} ({movie['year']})\n"
            f"Genres: {movie['genres']}\n"
            f"Rating: {movie['rating']}\n"
            f"Available on: {movie['available_on']}\n"
            f"Similarity score: {movie['similarity_score']}\n\n"
        )

    return answer


# =========================
# Run agent
# =========================

if __name__ == "__main__":
    print("Movie Recommendation Agent")
    print("Example: I liked Me Before You and I want a romantic movie")
    print("Example: I like the movie Avatar and I want action movie")
    print()

    user_input = input("Write your request: ")

    response = movie_agent(user_input)

    print("\nAgent response:")
    print(response)