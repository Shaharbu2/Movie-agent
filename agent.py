import os
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from flask import Flask, request, jsonify, Response

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
from scipy.sparse import hstack, csr_matrix


app = Flask(__name__)

# ==============================================================
# 1. LOAD DATA
# ==============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "movies_master.csv")

NEEDED_COLUMNS = [
    "title", "overview", "genres", "keywords", "available_on",
    "vote_average", "popularity", "runtime", "vote_count", "release_year",
    "Netflix", "Hulu", "Prime Video", "Disney+"
]

header_cols = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
usecols = [c for c in NEEDED_COLUMNS if c in header_cols]

df = pd.read_csv(DATA_PATH, usecols=usecols, nrows=50000)

for col in NEEDED_COLUMNS:
    if col not in df.columns:
        if col in ["Netflix", "Hulu", "Prime Video", "Disney+"]:
            df[col] = 0
        elif col in ["vote_average", "popularity", "runtime", "vote_count", "release_year"]:
            df[col] = 0
        elif col == "available_on":
            df[col] = "Not available in dataset"
        else:
            df[col] = ""

for col in ["title", "overview", "genres", "keywords"]:
    df[col] = df[col].fillna("").astype(str)

df["available_on"] = df["available_on"].fillna("Not available in dataset").astype(str)

for col in ["vote_average", "popularity", "runtime", "vote_count", "release_year"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

for col in ["Netflix", "Hulu", "Prime Video", "Disney+"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int8)

df["vote_average"] = df["vote_average"].astype(np.float32)
df["popularity"] = df["popularity"].astype(np.float32)
df["runtime"] = df["runtime"].astype(np.float32)
df["vote_count"] = df["vote_count"].astype(np.float32)
df["release_year"] = df["release_year"].astype(np.int16)


# ==============================================================
# 2. HELPERS
# ==============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\u0590-\u05FF\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_items(x):
    return [i.strip() for i in str(x).split(",") if i.strip()]


def vectorize_column(col, max_features=None):
    vec = CountVectorizer(
        tokenizer=split_items,
        token_pattern=None,
        binary=True,
        max_features=max_features,
        dtype=np.int8
    )
    return vec.fit_transform(df[col].astype(str))


def is_hebrew(text):
    return bool(re.search(r"[\u0590-\u05FF]", text))


# ==============================================================
# 3. FEATURE PREPARATION
# ==============================================================

numeric_features = ["vote_average", "popularity", "runtime", "vote_count"]

scaler = MinMaxScaler()
numeric_scaled = scaler.fit_transform(df[numeric_features]).astype(np.float32)
numeric_sparse = csr_matrix(numeric_scaled)

genres_vec = vectorize_column("genres", max_features=40)
keywords_vec = vectorize_column("keywords", max_features=50)

cluster_data = hstack([numeric_sparse, genres_vec, keywords_vec], format="csr")

kmeans = MiniBatchKMeans(
    n_clusters=5,
    random_state=42,
    n_init=2,
    batch_size=2048
)

df["cluster"] = kmeans.fit_predict(cluster_data).astype(np.int8)

CLUSTER_NAMES = {
    0: "דרמה / פשע / היסטוריה",
    1: "קומדיה / רומנטיקה",
    2: "אקשן / מדע בדיוני / מתח",
    3: "משפחה / אנימציה / פנטזיה",
    4: "אימה / מסתורין / מתח"
}

overview_clean_list = [clean_text(x) for x in df["overview"].tolist()]

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=2000,
    ngram_range=(1, 2),
    dtype=np.float32
)

tfidf_matrix = tfidf.fit_transform(overview_clean_list)

sim_data_sparse = hstack([numeric_sparse, genres_vec, keywords_vec], format="csr")


# ==============================================================
# 4. ANOMALY DETECTION
# ==============================================================

iso_features = ["popularity", "vote_average", "vote_count", "runtime"]

iso_scaler = MinMaxScaler()
iso_scaled = iso_scaler.fit_transform(df[iso_features]).astype(np.float32)

iso = IsolationForest(
    n_estimators=20,
    max_samples=4096,
    contamination=0.05,
    random_state=42
)

df["anomaly"] = iso.fit_predict(iso_scaled).astype(np.int8)
df["anomaly_score"] = iso.decision_function(iso_scaled).astype(np.float32)


# ==============================================================
# 5. INTENT AND FILTERS
# ==============================================================

GENRE_KEYWORD_MAP = {
    "action": "Action", "אקשן": "Action",
    "adventure": "Adventure", "הרפתקה": "Adventure",
    "animation": "Animation", "אנימציה": "Animation",
    "comedy": "Comedy", "קומדיה": "Comedy", "funny": "Comedy", "מצחיק": "Comedy",
    "crime": "Crime", "פשע": "Crime",
    "documentary": "Documentary", "דוקומנטרי": "Documentary",
    "drama": "Drama", "דרמה": "Drama", "emotional": "Drama", "מרגש": "Drama",
    "family": "Family", "משפחה": "Family", "kids": "Family", "ילדים": "Family",
    "fantasy": "Fantasy", "פנטזיה": "Fantasy",
    "history": "History", "היסטוריה": "History",
    "war": "War", "מלחמה": "War",
    "horror": "Horror", "אימה": "Horror", "scary": "Horror", "מפחיד": "Horror",
    "mystery": "Mystery", "מסתורין": "Mystery",
    "romance": "Romance", "romantic": "Romance", "רומנטי": "Romance",
    "רומנטיקה": "Romance", "אהבה": "Romance", "love": "Romance",
    "science fiction": "Science Fiction", "sci-fi": "Science Fiction",
    "מדע בדיוני": "Science Fiction",
    "thriller": "Thriller", "מתח": "Thriller"
}

GENRE_TO_CLUSTER = {
    "Action": 2, "Adventure": 2, "Science Fiction": 2, "Thriller": 2,
    "Comedy": 1, "Romance": 1,
    "Drama": 0, "Crime": 0, "History": 0, "War": 0, "Documentary": 0,
    "Family": 3, "Animation": 3, "Fantasy": 3,
    "Horror": 4, "Mystery": 4
}

PLATFORM_PATTERNS = {
    "Netflix": ["netflix", "נטפליקס"],
    "Hulu": ["hulu", "הולו"],
    "Prime Video": ["prime", "amazon", "אמזון", "פריים"],
    "Disney+": ["disney", "disney+", "דיסני"]
}

MOVIE_WORDS = [
    "movie", "movies", "film", "films", "cinema", "recommend", "similar", "like",
    "netflix", "hulu", "prime", "disney", "genre", "rating", "year", "anomaly",
    "cluster", "action", "romance", "comedy", "horror", "drama", "thriller",
    "סרט", "סרטים", "קולנוע", "תמליץ", "המלצה", "דומה", "כמו",
    "נטפליקס", "דיסני", "פריים", "הולו", "זאנר", "ז׳אנר", "דירוג", "שנה",
    "חריג", "חריגות", "קלאסטר", "קבוצה", "אקשן", "רומנטי", "קומדיה",
    "אימה", "דרמה", "מתח"
]


def extract_genres(text):
    t = text.lower()
    matched = set()

    for kw in sorted(GENRE_KEYWORD_MAP, key=len, reverse=True):
        if kw in t:
            matched.add(GENRE_KEYWORD_MAP[kw])

    return list(matched)


def genres_to_clusters(genres):
    return {GENRE_TO_CLUSTER[g] for g in genres if g in GENRE_TO_CLUSTER}


def extract_year(text):
    match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    return int(match.group(1)) if match else None


def extract_platform(text):
    t = text.lower()

    for platform, patterns in PLATFORM_PATTERNS.items():
        if any(p in t for p in patterns):
            return platform

    return None


def is_movie_related(text):
    t = text.lower().strip()

    if extract_year(t) or extract_platform(t) or extract_genres(t):
        return True

    return any(word in t for word in MOVIE_WORDS)


def detect_intent(text):
    t = text.lower()

    similar_patterns = [
        r"similar to (.+)",
        r"movies like (.+)",
        r"like the movie (.+)",
        r"like (.+)",
        r"דומה ל(.+)",
        r"סרטים כמו (.+)",
        r"כמו (.+)"
    ]

    for pattern in similar_patterns:
        match = re.search(pattern, t)
        if match:
            movie_title = match.group(1).strip().rstrip("?.,!")
            movie_title = re.split(r"\b(and|with|from|on)\b|ו|עם|משנת|בנטפליקס|בדיסני", movie_title)[0].strip()
            return "similar", movie_title

    if any(k in t for k in ["anomaly", "unusual", "outlier", "hidden gem", "underrated",
                            "חריג", "חריגות", "מוזר", "יהלום נסתר", "מוערך בחסר"]):
        return "anomaly", text

    if any(k in t for k in ["cluster", "clusters", "קלאסטר", "קלאסטרים", "קבוצה", "קבוצות"]):
        return "cluster_info", text

    return "search", text


def apply_filters(base_df, text):
    filtered = base_df.copy()

    year = extract_year(text)
    platform = extract_platform(text)

    if year is not None:
        filtered = filtered[filtered["release_year"] >= year]

    if platform is not None and platform in filtered.columns:
        filtered = filtered[filtered[platform] == 1]

    return filtered, year, platform


def get_streaming(row):
    platforms = []

    if row.get("Netflix", 0) == 1:
        platforms.append("Netflix")
    if row.get("Hulu", 0) == 1:
        platforms.append("Hulu")
    if row.get("Prime Video", 0) == 1:
        platforms.append("Prime Video")
    if row.get("Disney+", 0) == 1:
        platforms.append("Disney+")

    return ", ".join(platforms)


def row_to_result(rank, idx, score=0):
    row = df.loc[int(idx)]
    overview = str(row["overview"])

    return {
        "rank": rank,
        "title": row["title"],
        "year": int(row["release_year"]) if row["release_year"] else "N/A",
        "genres": row["genres"],
        "rating": round(float(row["vote_average"]), 1),
        "overview": overview[:170] + "..." if len(overview) > 170 else overview,
        "score": round(float(score), 3),
        "streaming": get_streaming(row)
    }


def no_relevant_answer():
    return {
        "intent": "no_answer",
        "reply": "",
        "results": []
    }


# ==============================================================
# 6. HANDLERS
# ==============================================================

def handle_search(user_text, top_n=3):
    if not is_movie_related(user_text):
        return no_relevant_answer()

    filtered, year, platform = apply_filters(df, user_text)

    matched_genres = extract_genres(user_text)
    matched_clusters = genres_to_clusters(matched_genres)

    if matched_genres:
        pattern = "|".join(matched_genres)
        filtered = filtered[filtered["genres"].str.contains(pattern, case=False, na=False)]

    if filtered.empty:
        return {
            "intent": "search",
            "reply": "",
            "results": [],
            "year": year,
            "platform": platform,
            "genres": matched_genres
        }

    cleaned = clean_text(user_text)
    user_vec = tfidf.transform([cleaned])
    tfidf_scores_all = cosine_similarity(user_vec, tfidf_matrix).flatten()

    indices = filtered.index.to_numpy()

    def genre_score(genres_text):
        if not matched_genres:
            return 0.0

        movie_genres = [g.strip() for g in str(genres_text).split(",")]
        return sum(1 for g in matched_genres if g in movie_genres) / max(len(matched_genres), 1)

    genre_scores = filtered["genres"].apply(genre_score).values
    cluster_scores = filtered["cluster"].apply(
        lambda c: 1.0 if c in matched_clusters else 0.0
    ).values

    tfidf_scores = tfidf_scores_all[indices]

    combined = (
        0.50 * tfidf_scores +
        0.30 * genre_scores +
        0.20 * cluster_scores
    )

    has_clear_filter = bool(year or platform or matched_genres)

    if len(combined) == 0:
        return no_relevant_answer()

    best_score = float(combined.max())

    if not has_clear_filter and best_score < 0.08:
        return no_relevant_answer()

    if has_clear_filter and best_score == 0:
        ranked = filtered.sort_values(
            ["vote_average", "vote_count", "popularity"],
            ascending=False
        ).head(top_n)

        if ranked.empty:
            return no_relevant_answer()

        results = [
            row_to_result(i + 1, idx, row["vote_average"] / 10)
            for i, (idx, row) in enumerate(ranked.iterrows())
        ]

    else:
        order = combined.argsort()[-top_n:][::-1]

        chosen = [
            pos for pos in order
            if combined[pos] >= max(0.04, best_score * 0.25)
        ]

        if not chosen:
            return no_relevant_answer()

        results = [
            row_to_result(i + 1, indices[pos], combined[pos])
            for i, pos in enumerate(chosen[:top_n])
        ]

    return {
        "intent": "search",
        "reply": "",
        "results": results,
        "year": year,
        "platform": platform,
        "genres": matched_genres
    }


def handle_similar(movie_title, user_text, top_n=3):
    matches = df[df["title"].str.lower().str.contains(movie_title.lower(), na=False)]

    if matches.empty:
        return {
            "intent": "similar",
            "reply": "",
            "results": []
        }

    idx = int(matches.index[0])

    scores = cosine_similarity(sim_data_sparse[idx], sim_data_sparse).flatten()
    scores[idx] = -1

    candidates = df.copy()
    candidates["similarity_score"] = scores
    candidates = candidates[candidates.index != idx]

    candidates, year, platform = apply_filters(candidates, user_text)

    matched_genres = extract_genres(user_text)

    if matched_genres:
        pattern = "|".join(matched_genres)
        candidates = candidates[candidates["genres"].str.contains(pattern, case=False, na=False)]

    if candidates.empty:
        return {
            "intent": "similar",
            "reply": "",
            "results": [],
            "year": year,
            "platform": platform,
            "genres": matched_genres
        }

    candidates = candidates[candidates["similarity_score"] >= 0.05]

    if candidates.empty:
        return {
            "intent": "similar",
            "reply": "",
            "results": [],
            "year": year,
            "platform": platform,
            "genres": matched_genres
        }

    top_results = candidates.sort_values("similarity_score", ascending=False).head(top_n)

    results = [
        row_to_result(i + 1, idx, row["similarity_score"])
        for i, (idx, row) in enumerate(top_results.iterrows())
    ]

    return {
        "intent": "similar",
        "reply": "",
        "results": results,
        "year": year,
        "platform": platform,
        "genres": matched_genres
    }


def handle_anomaly(user_text, top_n=3):
    filtered, year, platform = apply_filters(df, user_text)

    matched_genres = extract_genres(user_text)
    if matched_genres:
        pattern = "|".join(matched_genres)
        filtered = filtered[filtered["genres"].str.contains(pattern, case=False, na=False)]

    anomalies = filtered[filtered["anomaly"] == -1].copy()

    if anomalies.empty:
        return {
            "intent": "anomaly",
            "reply": "",
            "results": [],
            "year": year,
            "platform": platform,
            "genres": matched_genres
        }

    t = user_text.lower()

    if any(k in t for k in ["hidden gem", "underrated", "יהלום נסתר", "מוערך בחסר"]):
        subset = anomalies[
            (anomalies["vote_average"] >= 7.5) &
            (anomalies["vote_count"] < anomalies["vote_count"].quantile(0.45))
        ]
    elif any(k in t for k in ["flop", "פלופ"]):
        subset = anomalies[
            (anomalies["vote_average"] < 5.5) &
            (anomalies["popularity"] > anomalies["popularity"].median())
        ]
    elif any(k in t for k in ["long", "short", "runtime", "ארוך", "קצר"]):
        subset = anomalies[
            (anomalies["runtime"] > 180) |
            (anomalies["runtime"] < 60)
        ]
    else:
        subset = anomalies.sort_values("anomaly_score").head(top_n)

    if subset.empty:
        return {
            "intent": "anomaly",
            "reply": "",
            "results": [],
            "year": year,
            "platform": platform,
            "genres": matched_genres
        }

    subset = subset.head(top_n)

    results = [
        row_to_result(i + 1, idx, row["anomaly_score"])
        for i, (idx, row) in enumerate(subset.iterrows())
    ]

    return {
        "intent": "anomaly",
        "reply": "",
        "results": results,
        "year": year,
        "platform": platform,
        "genres": matched_genres
    }


def handle_cluster_info():
    summary = []

    for c, name in CLUSTER_NAMES.items():
        subset = df[df["cluster"] == c]
        all_genres = ", ".join(subset["genres"].dropna()).split(", ")
        top_genres = [g for g, _ in Counter(all_genres).most_common(3) if g]

        summary.append({
            "cluster": int(c),
            "name": name,
            "count": int(len(subset)),
            "top_genres": ", ".join(top_genres),
            "avg_rating": round(float(subset["vote_average"].mean()), 2)
        })

    return {
        "intent": "cluster_info",
        "reply": "",
        "clusters": summary,
        "results": []
    }


# ==============================================================
# 7. OPENAI RESPONSE
# ==============================================================

def call_openai(user_text, result):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return fallback_reply(user_text, result)

    try:
        import urllib.request

        lang = "Hebrew" if is_hebrew(user_text) else "English"

        results = result.get("results", [])
        intent = result.get("intent", "search")

        movies_text = ""

        if results:
            for r in results[:3]:
                streaming = r.get("streaming") or "Not available in streaming dataset"
                movies_text += (
                    f"- {r['title']} ({r['year']}), "
                    f"genres: {r['genres']}, "
                    f"rating: {r['rating']}/10, "
                    f"streaming: {streaming}, "
                    f"score: {r['score']}\n"
                )
        else:
            movies_text = "No matching movies were found in the dataset."

        system_prompt = (
            "You are a movie recommendation agent only. "
            "You must answer only questions related to movies, genres, movie recommendations, "
            "streaming platforms, ratings, similarity, clustering or anomaly detection in the movie dataset. "
            "Do not answer weather, politics, health, general knowledge or unrelated questions. "
            "Use only the provided dataset results. "
            "Do not invent movies. "
            "If no results were found, politely say that no suitable movies were found and suggest changing the year, genre or platform. "
            "Do not write duplicate sections. "
            "Do not say 'here are three movies' if there are no results. "
            "If there are results, write a short friendly explanation and refer to the movies below."
        )

        user_prompt = (
            f"Answer language: {lang}\n"
            f"User question: {user_text}\n"
            f"Detected intent: {intent}\n"
            f"Detected year: {result.get('year')}\n"
            f"Detected platform: {result.get('platform')}\n"
            f"Detected genres: {result.get('genres')}\n\n"
            f"Dataset results:\n{movies_text}\n\n"
            "Write one concise answer. "
            "If there are movie results, mention the recommended titles briefly. "
            "If there are no results, explain that no matching movies were found. "
            "Do not add movies that are not listed."
        )

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 250,
            "temperature": 0.4
        }

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + api_key
            },
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read())
            return data["choices"][0]["message"]["content"]

    except Exception as e:
        print("OpenAI error:", str(e))
        return fallback_reply(user_text, result)


def fallback_reply(user_text, result):
    heb = is_hebrew(user_text)
    results = result.get("results", [])

    if not results:
        if heb:
            return "לא מצאתי סרטים שמתאימים בדיוק לבקשה שלך. אפשר לנסות לשנות שנה, ז׳אנר או פלטפורמת צפייה."
        return "I could not find movies that match your request. Try changing the year, genre, or streaming platform."

    titles = ", ".join([r["title"] for r in results[:3]])

    if heb:
        return f"מצאתי כמה סרטים שמתאימים לבקשה שלך: {titles}."
    return f"I found a few movies that match your request: {titles}."


# ==============================================================
# 8. ROUTES
# ==============================================================

@app.route("/health")
def health():
    return jsonify({"status": "ok", "movies": int(len(df))})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_text = data.get("message", "").strip()

    if not user_text:
        return jsonify({
            "reply": "תכתבי לי איזה סרט או סוג סרט את מחפשת 🎬",
            "results": []
        })

    if not is_movie_related(user_text):
        result = {
            "intent": "out_of_scope",
            "reply": "",
            "results": []
        }
        result["reply"] = call_openai(user_text, result)
        return jsonify(result)

    intent, payload = detect_intent(user_text)

    if intent == "similar":
        result = handle_similar(payload, user_text)
    elif intent == "anomaly":
        result = handle_anomaly(user_text)
    elif intent == "cluster_info":
        result = handle_cluster_info()
    else:
        result = handle_search(user_text)

    result["reply"] = call_openai(user_text, result)

    return jsonify(result)


@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


# ==============================================================
# 9. HTML
# ==============================================================

HTML_PAGE = """
<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<title>צ׳אטבוט סרטים</title>
<style>
body {
    font-family: Arial, sans-serif;
    background: #111;
    color: white;
    margin: 0;
    padding: 0;
}
.container {
    width: min(900px, 92vw);
    margin: 40px auto;
}
h1 {
    color: #ffcc66;
}
#chat {
    background: #fff;
    color: #222;
    height: 460px;
    overflow-y: auto;
    padding: 20px;
    border-radius: 16px;
}
.msg {
    margin: 12px 0;
    line-height: 1.6;
}
.user {
    text-align: left;
}
.bot {
    text-align: right;
}
.bubble {
    display: inline-block;
    padding: 12px 16px;
    border-radius: 16px;
    max-width: 80%;
}
.user .bubble {
    background: #d71920;
    color: white;
}
.bot .bubble {
    background: #eeeeee;
    color: #222;
}
.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 12px;
    margin-top: 12px;
}
.card {
    background: white;
    color: #222;
    padding: 14px;
    border-radius: 14px;
    border: 1px solid #ddd;
}
.card-title {
    font-weight: bold;
    color: #b10e15;
}
.meta {
    font-size: 13px;
    color: #555;
}
.input-row {
    display: flex;
    gap: 10px;
    margin-top: 14px;
}
input {
    flex: 1;
    padding: 14px;
    border-radius: 12px;
    border: none;
    font-size: 16px;
}
button {
    padding: 14px 24px;
    border: none;
    border-radius: 12px;
    background: #d71920;
    color: white;
    font-weight: bold;
    cursor: pointer;
}
</style>
</head>
<body>
<div class="container">
    <h1>🎬 צ׳אטבוט המלצות סרטים</h1>
    <p>אפשר לשאול על סרטים לפי שנה, ז׳אנר, פלטפורמה, סרטים דומים, קלאסטרים או חריגות.</p>

    <div id="chat">
        <div class="msg bot">
            <div class="bubble">שלום! כתבו לי איזה סרט או סוג סרט אתם מחפשים 🍿</div>
        </div>
    </div>

    <div class="input-row">
        <input id="inp" placeholder="לדוגמה: אהבתי Avatar ואני רוצה סרטי אקשן מ-2021 ומעלה">
        <button onclick="send()">שליחה</button>
    </div>
</div>

<script>
const chat = document.getElementById("chat");
const inp = document.getElementById("inp");

function esc(s) {
    return String(s || "").replace(/[&<>"']/g, m => ({
        "&":"&amp;",
        "<":"&lt;",
        ">":"&gt;",
        '"':"&quot;",
        "'":"&#39;"
    }[m]));
}

function add(role, html) {
    const div = document.createElement("div");
    div.className = "msg " + role;
    div.innerHTML = html;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

function cards(results) {
    if (!results || !results.length) return "";

    let html = '<div class="cards">';

    results.forEach(r => {
        html += `
        <div class="card">
            <div class="card-title">${esc(r.rank)}. ${esc(r.title)}</div>
            <div class="meta">${esc(r.year)} | ⭐ ${esc(r.rating)}/10 | התאמה ${esc(r.score)}</div>
            <div><b>ז׳אנרים:</b> ${esc(r.genres)}</div>
            ${r.streaming ? `<div><b>זמין ב:</b> ${esc(r.streaming)}</div>` : ""}
            <div>${esc(r.overview)}</div>
        </div>`;
    });

    html += "</div>";
    return html;
}

function send() {
    const text = inp.value.trim();
    if (!text) return;

    add("user", '<div class="bubble">' + esc(text) + '</div>');
    inp.value = "";

    fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message: text})
    })
    .then(r => r.json())
    .then(data => {
        add("bot", '<div class="bubble">' + esc(data.reply) + '</div>' + cards(data.results));
    })
    .catch(() => {
        add("bot", '<div class="bubble">משהו השתבש. נסו שוב.</div>');
    });
}

inp.addEventListener("keydown", e => {
    if (e.key === "Enter") send();
});
</script>
</body>
</html>
"""


# ==============================================================
# 10. RUN
# ==============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
