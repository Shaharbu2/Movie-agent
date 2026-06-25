import os

import re

import gc

import json

import difflib

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

# 1. LOAD & PREPARE DATA - optimized for Render free memory

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

dtype_map = {

    "vote_average": np.float32, "popularity": np.float32,

    "runtime": np.float32, "vote_count": np.float32,

    "release_year": np.float32,

    "Netflix": np.int8, "Hulu": np.int8,

    "Prime Video": np.int8, "Disney+": np.int8,

}

usable_dtypes = {k: v for k, v in dtype_map.items() if k in header_cols}

df = pd.read_csv(DATA_PATH, usecols=usecols, nrows=20000, dtype=usable_dtypes)



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



df["title"] = df["title"].fillna("").astype(str)

df["overview"] = df["overview"].fillna("").astype(str)

df["genres"] = df["genres"].fillna("").astype(str)

df["keywords"] = df["keywords"].fillna("").astype(str)

df["available_on"] = df["available_on"].fillna("Not available in dataset").astype(str)



for c in ["vote_average", "popularity", "runtime", "vote_count"]:

    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.float32)



df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(np.int16)



for col in ["Netflix", "Hulu", "Prime Video", "Disney+"]:

    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int8)



# ==============================================================

# 2. HELPERS

# ==============================================================



def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-z0-9\u0590-\u05FF\s]", " ", text)

    return re.sub(r"\s+", " ", text).strip()





def clean_title(text):

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

    return bool(re.search(r"[\u0590-\u05FF]", str(text)))





def normalize_hebrew_typos(text):

    text = str(text)

    replacements = {

        "זאנר": "ז׳אנר",

        "ז'אנר": "ז׳אנר",

        "נטפליס": "נטפליקס",

        "נטפליקסס": "נטפליקס",

        "דיסניי": "דיסני",

        "אקשין": "אקשן",

        "קומדייה": "קומדיה",

        "סרטימ": "סרטים",

        "מומלצ": "מומלץ",

        "קומדיא": "קומדיה", "איימה": "אימה", "רומנתי": "רומנטי", "דרמא": "דרמה",

        "נטפליכס": "נטפליקס", "דסני": "דיסני", "סרת": "סרט",

    }

    for wrong, right in replacements.items():

        text = text.replace(wrong, right)

    return text





EN_TYPO_REPLACEMENTS = {
    "romcom": "romantic comedy", "rommance": "romance", "romace": "romance",
    "comedey": "comedy", "commedy": "comedy", "comdy": "comedy",
    "horrer": "horror", "horor": "horror",
    "actoin": "action", "acion": "action",
    "thriler": "thriller", "triller": "thriller",
    "netfix": "netflix", "netflx": "netflix", "netfli": "netflix",
    "disny": "disney", "diney": "disney",
    "amazom": "amazon", "prme": "prime",
}


def normalize_english_typos(text):
    text = str(text)
    for wrong, right in EN_TYPO_REPLACEMENTS.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.IGNORECASE)
    return text


def fuzzy_normalize_keywords(text):
    raw = str(text)
    tokens = re.findall(r"[a-zA-Z]+|[\u0590-\u05FF]+", raw)
    dictionary = list(GENRE_KEYWORD_MAP.keys()) + [p for pats in PLATFORM_PATTERNS.values() for p in pats] + MOVIE_WORDS
    fixed = raw
    for tok in tokens:
        if len(tok) < 4:
            continue
        matches = difflib.get_close_matches(tok.lower(), dictionary, n=1, cutoff=0.82)
        if matches:
            fixed = re.sub(rf"(?<!\w){re.escape(tok)}(?!\w)", matches[0], fixed, flags=re.IGNORECASE)
    return fixed


def normalize_user_text(text):
    text = normalize_hebrew_typos(str(text).strip())
    text = normalize_english_typos(text)
    try:
        text = fuzzy_normalize_keywords(text)
    except NameError:
        pass
    return text



# Fast title lookup helpers

df["title_clean"] = df["title"].apply(clean_title)

TITLE_CLEAN_LIST = df["title_clean"].dropna().astype(str).unique().tolist()



# ==============================================================

# 3. CLUSTERING - sparse

# ==============================================================



numeric_features = ["vote_average", "popularity", "runtime", "vote_count"]

scaler = MinMaxScaler()

numeric_scaled = scaler.fit_transform(df[numeric_features]).astype(np.float32)

numeric_sparse = csr_matrix(numeric_scaled)



genres_vec = vectorize_column("genres", max_features=20)

keywords_vec = vectorize_column("keywords", max_features=20)



cluster_data = hstack([numeric_sparse, genres_vec, keywords_vec], format="csr")



kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, n_init=2, batch_size=2048)

df["cluster"] = kmeans.fit_predict(cluster_data).astype(np.int8)



# Build sim_data BEFORE freeing numeric_sparse

sim_data_sparse = hstack([numeric_sparse, genres_vec, keywords_vec], format="csr")



del cluster_data, numeric_scaled, numeric_sparse, genres_vec, keywords_vec

gc.collect()



CLUSTER_NAMES = {

    0: "דרמה / פשע / היסטוריה",

    1: "קומדיה / רומנטיקה",

    2: "אקשן / מדע בדיוני / מתח",

    3: "משפחה / אנימציה / פנטזיה",

    4: "אימה / מסתורין / מתח"

}



# ==============================================================

# 4. TF-IDF

# ==============================================================



overview_clean_list = [clean_text(x) for x in df["overview"].tolist()]

tfidf = TfidfVectorizer(stop_words="english", max_features=800, ngram_range=(1, 2), dtype=np.float32)

tfidf_matrix = tfidf.fit_transform(overview_clean_list)

del overview_clean_list  # free the list immediately after fitting



# ==============================================================

# 5. ANOMALY DETECTION - light

# ==============================================================



iso_features = ["popularity", "vote_average", "vote_count", "runtime"]

iso_scaler = MinMaxScaler()

iso_scaled = iso_scaler.fit_transform(df[iso_features]).astype(np.float32)

iso = IsolationForest(n_estimators=10, max_samples=2048, contamination=0.05, random_state=42)

df["anomaly"] = iso.fit_predict(iso_scaled).astype(np.int8)

df["anomaly_score"] = iso.decision_function(iso_scaled).astype(np.float32)

del iso_scaled; gc.collect()



# ==============================================================

# 6. INTENT + FILTER DETECTION

# ==============================================================



SMALLTALK_PATTERNS = [

    "hi", "hello", "hey", "good morning", "good evening", "how are you",

    "היי", "הי", "שלום", "מה קורה", "מה נשמע", "מה שלומך",

    "בוקר טוב", "ערב טוב", "צהריים טובים"

]



GENRE_KEYWORD_MAP = {

    "action": "Action", "אקשן": "Action", "fight": "Action", "battle": "Action",

    "adventure": "Adventure", "הרפתקה": "Adventure",

    "animation": "Animation", "אנימציה": "Animation", "cartoon": "Animation",

    "comedy": "Comedy", "קומדיה": "Comedy", "funny": "Comedy", "מצחיק": "Comedy",

    "crime": "Crime", "פשע": "Crime", "detective": "Crime",

    "documentary": "Documentary", "דוקומנטרי": "Documentary",

    "drama": "Drama", "דרמה": "Drama", "emotional": "Drama", "מרגש": "Drama",

    "family": "Family", "משפחה": "Family", "kids": "Family", "ילדים": "Family",

    "fantasy": "Fantasy", "פנטזיה": "Fantasy", "magic": "Fantasy", "קסם": "Fantasy",

    "history": "History", "היסטוריה": "History", "war": "War", "מלחמה": "War",

    "horror": "Horror", "אימה": "Horror", "scary": "Horror", "מפחיד": "Horror",

    "mystery": "Mystery", "מסתורין": "Mystery",

    "romance": "Romance", "romantic": "Romance", "רומנטי": "Romance",

    "love": "Romance", "אהבה": "Romance", "זוגי": "Romance",

    "science fiction": "Science Fiction", "sci-fi": "Science Fiction", "מדע בדיוני": "Science Fiction",

    "thriller": "Thriller", "מתח": "Thriller",

}



GENRE_TO_CLUSTER = {

    "Action": 2, "Adventure": 2, "Science Fiction": 2, "Thriller": 2,

    "Comedy": 1, "Romance": 1,

    "Drama": 0, "Crime": 0, "History": 0, "War": 0, "Documentary": 0,

    "Family": 3, "Animation": 3, "Fantasy": 3,

    "Horror": 4, "Mystery": 4,

}



PLATFORM_PATTERNS = {

    "Netflix": ["netflix", "נטפליקס"],

    "Hulu": ["hulu", "הולו"],

    "Prime Video": ["prime", "amazon", "אמזון", "פריים"],

    "Disney+": ["disney", "דיסני", "disney+"],

}



MOVIE_WORDS = [

    "סרט", "סרטים", "קולנוע", "נטפליקס", "דיסני", "פריים", "הולו",

    "קומדיה", "אקשן", "אימה", "דרמה", "רומנטי", "רומנטיקה", "מתח",

    "אנימציה", "ילדים", "פנטזיה", "מדע בדיוני", "דומה", "כמו",

    "המלצה", "תמליץ", "לראות", "צפייה", "דירוג", "שנה",

    "movie", "movies", "film", "films", "cinema", "netflix", "hulu",

    "prime", "disney", "comedy", "action", "horror", "drama", "romance",

    "thriller", "animation", "similar", "like", "recommend", "rating", "year"

]





def is_smalltalk(text):

    t = clean_text(text)

    if not t:

        return True

    return any(p in t for p in SMALLTALK_PATTERNS) and not any(w in t for w in MOVIE_WORDS)





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

    # \b fails on "מ2006" (Hebrew letter before digit), so use lookahead/lookbehind instead

    m = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", text)

    return int(m.group(1)) if m else None





def extract_platform(text):

    t = text.lower()

    for platform, pats in PLATFORM_PATTERNS.items():

        if any(p in t for p in pats):

            return platform

    return None





def find_movie_title(user_text):

    """Find a movie title in user input with exact and fuzzy matching."""

    text_clean = clean_title(user_text)



    # Vectorized exact containment: prefer longest matched title

    mask = df["title_clean"].apply(lambda tc: bool(tc) and tc in text_clean)

    hits = df[mask]

    if not hits.empty:

        best_idx = hits["title_clean"].str.len().idxmax()

        return df.loc[best_idx, "title"]



    # Try extracting after common phrases

    patterns = [

        r"similar to (.+)",

        r"movies like (.+)",

        r"like the movie (.+)",

        r"i liked (.+)",

        r"i like (.+)",

        r"דומה ל(.+)",

        r"סרטים כמו (.+)",

        r"כמו (.+)",

        r"אהבתי את (.+)",

        r"אהבתי (.+)"

    ]



    candidate = None

    low = user_text.lower()

    for pat in patterns:

        m = re.search(pat, low)

        if m:

            candidate = m.group(1)

            candidate = re.split(

                r"\b(and|with|from|on|that|which|for)\b|ו|עם|משנת|מ|בנטפליקס|בדיסני|שיהיה|שרוצה",

                candidate

            )[0].strip(" ?.,!")

            break



    if not candidate:

        return None



    candidate_clean = clean_title(candidate)

    matches = difflib.get_close_matches(candidate_clean, TITLE_CLEAN_LIST, n=1, cutoff=0.72)



    if matches:

        matched_clean = matches[0]

        hit = df[df["title_clean"] == matched_clean]

        if not hit.empty:

            return hit.iloc[0]["title"]



    return None





def is_movie_related(text):

    t = clean_text(text)

    if is_smalltalk(t):

        return True

    if extract_year(t) or extract_platform(t) or extract_genres(t):

        return True

    if find_movie_title(text):

        return True

    return any(w in t for w in MOVIE_WORDS)





def detect_intent(text):

    t = clean_text(text)



    if is_smalltalk(text):

        return "smalltalk", text



    if any(k in t for k in [

        "unusual", "anomaly", "outlier", "flop", "hidden gem", "underrated",

        "פלופ", "יהלום נסתר", "חריג", "חריגות", "מוזר", "מוערך בחסר",

        "ארוך", "קצר"

    ]):

        return "anomaly", text



    if any(k in t for k in ["cluster", "clusters", "קלסטר", "קלאסטר", "קבוצה", "סוגי סרטים"]):

        return "cluster_info", text



    movie_title = find_movie_title(text)

    if movie_title and any(k in t for k in ["similar", "like", "דומה", "כמו", "אהבתי"]):

        return "similar", movie_title



    return "search", text





def apply_filters(base_df, text):

    filtered = base_df.copy()

    year = extract_year(text)

    platform = extract_platform(text)



    if year is not None:

        # Year means "from this year and above"

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

    overview = row["overview"]

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





def empty_result(intent="search", **extra):

    result = {"intent": intent, "reply": "", "results": []}

    result.update(extra)

    return result



# ==============================================================

# 7. HANDLERS - one result at a time

# ==============================================================



def handle_smalltalk(user_text):

    return empty_result("smalltalk")





def handle_out_of_scope(user_text):

    return empty_result("out_of_scope")





def handle_search(user_text, top_n=1, exclude_titles=None):

    if not is_movie_related(user_text):

        return handle_out_of_scope(user_text)



    filtered, year, platform = apply_filters(df, user_text)



    matched_genres = extract_genres(user_text)

    matched_clusters = genres_to_clusters(matched_genres)



    if matched_genres:

        pattern = "|".join(matched_genres)

        filtered = filtered[filtered["genres"].str.contains(pattern, case=False, na=False)]



    if filtered.empty:

        return empty_result("search", year=year, platform=platform, genres=matched_genres)



    exclude_set = {str(t).lower() for t in (exclude_titles or [])}

    if exclude_set:

        filtered = filtered[~filtered["title"].str.lower().isin(exclude_set)]

        if filtered.empty:

            return empty_result("search", year=year, platform=platform, genres=matched_genres, exhausted=True)



    cleaned = clean_text(user_text)



    user_vec = tfidf.transform([cleaned])

    tfidf_scores_all = cosine_similarity(user_vec, tfidf_matrix).flatten()



    indices = filtered.index.to_numpy()



    def genre_score(gs):

        if not matched_genres:

            return 0.0

        movie_genres = [g.strip() for g in str(gs).split(",")]

        return sum(1 for g in matched_genres if g in movie_genres) / max(len(matched_genres), 1)



    genre_scores = filtered["genres"].apply(genre_score).values

    cluster_scores = filtered["cluster"].apply(lambda c: 1.0 if c in matched_clusters else 0.0).values

    tfidf_scores = tfidf_scores_all[indices]



    combined = 0.50 * tfidf_scores + 0.30 * genre_scores + 0.20 * cluster_scores



    has_clear_filter = bool(year or platform or matched_genres)

    best_score = float(combined.max()) if len(combined) else 0.0



    if not has_clear_filter and best_score < 0.08:

        return empty_result("search", year=year, platform=platform, genres=matched_genres)



    # With filters but no TF-IDF signal: only return if genre/cluster score is meaningful

    if has_clear_filter and best_score < 0.05 and not matched_genres:

        return empty_result("search", year=year, platform=platform, genres=matched_genres)



    if has_clear_filter and best_score == 0 and matched_genres:

        # Genre filter matched rows but no TF-IDF overlap — rank by quality

        ranked = filtered.sort_values(["vote_average", "vote_count", "popularity"], ascending=False).head(top_n)

        if ranked.empty:

            return empty_result("search", year=year, platform=platform, genres=matched_genres)



        results = [

            row_to_result(i + 1, idx, round(float(row["vote_average"]) / 10, 3))

            for i, (idx, row) in enumerate(ranked.iterrows())

        ]

    else:

        order = combined.argsort()[-top_n:][::-1]

        chosen = [pos for pos in order if combined[pos] >= max(0.04, best_score * 0.25)]



        if not chosen:

            return empty_result("search", year=year, platform=platform, genres=matched_genres)



        results = [row_to_result(i + 1, indices[pos], combined[pos]) for i, pos in enumerate(chosen[:top_n])]



    return {

        "intent": "search",

        "reply": "",

        "results": results,

        "genres": matched_genres,

        "year": year,

        "platform": platform

    }





def handle_similar(movie_title, user_text, top_n=1, exclude_titles=None):

    matches = df[df["title"].str.lower() == str(movie_title).lower()]



    if matches.empty:

        matches = df[df["title"].str.lower().str.contains(str(movie_title).lower(), na=False)]



    if matches.empty:

        return empty_result("similar")



    idx = int(matches.index[0])

    found = df.loc[idx, "title"]



    # Compute similarity for just one row vs all — avoids building full NxN matrix

    scores = cosine_similarity(sim_data_sparse[idx:idx+1], sim_data_sparse).flatten()

    scores[idx] = -1



    candidates = df.copy()

    candidates["similarity_score"] = scores

    candidates = candidates[candidates.index != idx]



    candidates, year, platform = apply_filters(candidates, user_text)



    exclude_set = {str(t).lower() for t in (exclude_titles or [])}

    if exclude_set:

        candidates = candidates[~candidates["title"].str.lower().isin(exclude_set)]



    matched_genres = extract_genres(user_text)

    if matched_genres:

        pattern = "|".join(matched_genres)

        candidates = candidates[candidates["genres"].str.contains(pattern, case=False, na=False)]



    if candidates.empty:

        return empty_result("similar", year=year, platform=platform, genres=matched_genres, reference_movie=found)



    # Do not return weak/random results

    candidates = candidates[candidates["similarity_score"] >= 0.05]



    if candidates.empty:

        return empty_result("similar", year=year, platform=platform, genres=matched_genres, reference_movie=found)



    top_results = candidates.sort_values("similarity_score", ascending=False).head(top_n)



    results = [

        row_to_result(i + 1, tidx, row["similarity_score"])

        for i, (tidx, row) in enumerate(top_results.iterrows())

    ]



    return {

        "intent": "similar",

        "reply": "",

        "results": results,

        "genres": matched_genres,

        "year": year,

        "platform": platform,

        "reference_movie": found

    }





def handle_anomaly(user_text, top_n=1):

    filtered, year, platform = apply_filters(df, user_text)



    matched_genres = extract_genres(user_text)

    if matched_genres:

        pattern = "|".join(matched_genres)

        filtered = filtered[filtered["genres"].str.contains(pattern, case=False, na=False)]



    anomalies = filtered[filtered["anomaly"] == -1].copy()



    if anomalies.empty:

        return empty_result("anomaly", year=year, platform=platform, genres=matched_genres)



    t = clean_text(user_text)



    if any(k in t for k in ["flop", "פלופ", "budget", "תקציב"]):

        subset = anomalies[(anomalies["vote_average"] < 5.5) & (anomalies["popularity"] > anomalies["popularity"].median())]

        anomaly_type = "flop"

    elif any(k in t for k in ["hidden gem", "יהלום נסתר", "underrated", "מוערך בחסר"]):

        subset = anomalies[(anomalies["vote_average"] >= 7.5) & (anomalies["vote_count"] < anomalies["vote_count"].quantile(0.45))]

        anomaly_type = "hidden_gem"

    elif any(k in t for k in ["long", "ארוך", "runtime"]):

        subset = anomalies[anomalies["runtime"] > 180]

        anomaly_type = "long_runtime"

    elif any(k in t for k in ["short", "קצר", "runtime"]):

        subset = anomalies[anomalies["runtime"] < 60]

        anomaly_type = "short_runtime"

    else:

        subset = anomalies.sort_values("anomaly_score").head(top_n)

        anomaly_type = "general"



    if subset.empty:

        return empty_result("anomaly", year=year, platform=platform, genres=matched_genres, anomaly_type=anomaly_type)



    subset = subset.head(top_n)



    results = [

        row_to_result(i + 1, idx, row["anomaly_score"])

        for i, (idx, row) in enumerate(subset.iterrows())

    ]



    return {

        "intent": "anomaly",

        "reply": "",

        "results": results,

        "genres": matched_genres,

        "year": year,

        "platform": platform,

        "anomaly_type": anomaly_type

    }





def handle_cluster_info(user_text):

    summary = []

    for c, name in CLUSTER_NAMES.items():

        subset = df[df["cluster"] == c]

        all_g = ", ".join(subset["genres"].dropna()).split(", ")

        top_g = [g for g, _ in Counter(all_g).most_common(3) if g]

        summary.append({

            "cluster": int(c),

            "name": name,

            "count": int(len(subset)),

            "top_genres": ", ".join(top_g),

            "avg_rating": round(float(subset["vote_average"].mean()), 2)

        })



    return {"intent": "cluster_info", "reply": "", "clusters": summary, "results": []}



# ============================================================== 
# 8. CONVERSATION FLOW + OPENAI RESPONSE
# ============================================================== 

CONVERSATIONS = {}

# The agent no longer follows a rigid step-by-step form.
# It keeps a flexible state and asks only for the most useful missing detail.
RECOMMENDATION_SIGNALS = ["year", "reference_movie", "occasion", "platform"]


def new_state():
    return {
        "genre": None,
        "year": None,
        "era": None,
        "reference_movie": None,
        "occasion": None,
        "platform": None,
        "asked_fields": [],
        "shown_titles": [],
        "last_query": "",
        "done": False
    }


def get_conversation_key():
    return request.headers.get("X-Session-Id") or request.remote_addr or "default"


def wants_reset(text):
    t = clean_text(text)
    return any(x in t for x in [
        "restart", "reset", "start over", "start again", "can we start again", "new recommendation", "new movie",
        "התחל מחדש", "איפוס", "המלצה חדשה", "סרט חדש", "מהתחלה", "נתחיל מחדש"
    ])


def is_none_answer(text):
    t = clean_text(text)
    return t in ["no", "none", "skip", "dont know", "don't know", "no preference",
                 "לא", "אין", "דלג", "לא יודע", "לא יודעת", "אין העדפה"]


def wants_more(text):
    t = clean_text(text)
    return t in [
        "yes", "y", "yeah", "yep", "sure", "another", "more", "one more", "next",
        "כן", "כן עוד", "עוד", "אפשר עוד", "תן עוד", "עוד אחד", "סרט נוסף", "יאללה"
    ]


def maybe_out_of_scope_during_interview(text):
    """
    Guardrail for clearly unrelated questions.
    It catches short off-topic questions too, such as "מה המזג אוויר?".
    It does not block normal short movie answers like "קומדיה", "דייט", "חדשים", "Netflix".
    """
    t = clean_text(text)
    unrelated = [
        "weather", "forecast", "temperature", "rain", "salary", "excel", "politics",
        "news", "stock", "recipe", "football", "basketball", "bank", "tax",
        "מזג", "מזג אוויר", "תחזית", "גשם", "טמפרטורה", "שכר", "משכורת",
        "אקסל", "פוליטיקה", "חדשות", "מניות", "מתכון", "כדורגל", "כדורסל",
        "בנק", "מס", "מיסים", "גובה", "משקל"
    ]
    return any(x in t for x in unrelated)


def friendly_out_of_scope_reply(text):
    heb = is_hebrew(text)
    if heb:
        return "מצטער 😊 אני כאן כדי לעזור לך לבחור סרט 🎬 בוא נחזור לסרטים — איזה סוג סרט בא לך לראות?"
    return "Sorry 😊 I’m here to help you choose a movie 🎬 Let’s get back to movie night — what kind of film are you looking for?"


def smalltalk_reply(text, state):
    heb = is_hebrew(text)
    q = next_interview_question(state, heb)
    if heb:
        return f"הכול מצוין 😊 שמח שאתה כאן. {q}"
    return f"I’m doing great 😊 Glad you’re here. {q}"


def extract_era(text):
    t = clean_text(text)
    if any(x in t for x in ["new", "recent", "modern", "latest", "חדש", "חדשים", "מודרני", "מהשנים האחרונות"]):
        return "recent"
    if any(x in t for x in ["classic", "old", "older", "קלאסי", "קלאסיקה", "ישן", "ישנים", "קלאסיקות"]):
        return "classic"
    return None


def extract_occasion(text):
    t = clean_text(text)
    occasion_map = [
        ("date night", ["date", "romantic night", "דייט", "זוגי", "ערב זוגי"]),
        ("friends night", ["friends", "party", "pajama", "חברים", "מסיבה", "פיגמות", "ערב חברים"]),
        ("family watch", ["family", "kids", "children", "משפחה", "ילדים"]),
        ("solo watch", ["solo", "alone", "casual", "לבד", "עצמי", "סתם", "רגוע", "צפייה לבד"]),
        ("cozy night", ["cozy", "relaxing", "chill", "נעים", "רגוע", "בית", "ערב בבית"])
    ]
    for value, patterns in occasion_map:
        if any(p in t for p in patterns):
            return value
    return None


def merge_preferences_from_text(state, user_text):
    """
    Extract all useful movie preferences from a single user message.
    This is what makes the agent flexible instead of a rigid form.
    """
    if is_none_answer(user_text):
        return state

    genres = extract_genres(user_text)
    if genres and not state.get("genre"):
        state["genre"] = ", ".join(genres)

    year = extract_year(user_text)
    if year and not state.get("year"):
        state["year"] = str(year)

    era = extract_era(user_text)
    if era and not state.get("era"):
        state["era"] = era

    platform = extract_platform(user_text)
    if platform and not state.get("platform"):
        state["platform"] = platform

    occasion = extract_occasion(user_text)
    if occasion and not state.get("occasion"):
        state["occasion"] = occasion

    movie_title = find_movie_title(user_text)
    if movie_title and not state.get("reference_movie"):
        state["reference_movie"] = movie_title

    return state


def known_signal_count(state):
    count = 0
    if state.get("genre"):
        count += 1
    if state.get("year") or state.get("era"):
        count += 1
    if state.get("reference_movie"):
        count += 1
    if state.get("occasion"):
        count += 1
    if state.get("platform"):
        count += 1
    return count


def has_enough_for_recommendation(state):
    """
    Do not recommend after genre only.
    Recommend once we have genre/reference plus at least one more useful signal,
    or any 3 meaningful signals.
    """
    if state.get("reference_movie") and (state.get("genre") or state.get("occasion") or state.get("platform") or state.get("year") or state.get("era")):
        return True

    if state.get("genre"):
        extra = sum(bool(state.get(k)) for k in ["year", "era", "reference_movie", "occasion", "platform"])
        return extra >= 1

    return known_signal_count(state) >= 3


def next_missing_field(state):
    """
    Dynamic priority:
    - First get genre/vibe unless a reference movie already gives a clear direction.
    - Then ask the most natural missing question based on what is already known.
    """
    if not state.get("genre") and not state.get("reference_movie"):
        return "genre"

    if not state.get("occasion") and "occasion" not in state.get("asked_fields", []):
        return "occasion"

    if not state.get("platform") and "platform" not in state.get("asked_fields", []):
        return "platform"

    if not state.get("year") and not state.get("era") and "year" not in state.get("asked_fields", []):
        return "year"

    if not state.get("reference_movie") and "reference_movie" not in state.get("asked_fields", []):
        return "reference_movie"

    return None


def next_interview_question(state, heb=True):
    field = next_missing_field(state)
    if field:
        state.setdefault("asked_fields", [])
        if field not in state["asked_fields"]:
            state["asked_fields"].append(field)

    if heb:
        questions = {
            "genre": "איזה סגנון או וייב בא לך היום? למשל קומדיה, אימה, אקשן, רומנטי או משהו קליל.",
            "occasion": "נשמע טוב 🎬 לאיזו סיטואציה זה — דייט, חברים, משפחה או צפייה רגועה לבד?",
            "platform": "יש פלטפורמה מועדפת לצפייה? Netflix, Disney+, Prime Video, Hulu, או שאין העדפה?",
            "year": "בא לך משהו חדש יחסית או דווקא קלאסיקה ישנה?",
            "reference_movie": "יש סרט שאהבת לאחרונה והיית רוצה משהו באותו וייב? אם אין, לגמרי בסדר."
        }
        return questions.get(field, "מעולה, יש לי כיוון טוב. רוצה שאבחר לך סרט מתאים?")
    else:
        questions = {
            "genre": "What kind of vibe are you in the mood for today? Comedy, horror, action, romance, or something light?",
            "occasion": "Nice 🎬 What’s the occasion — date night, friends, family, or a relaxed solo watch?",
            "platform": "Do you have a preferred streaming platform? Netflix, Disney+, Prime Video, Hulu, or no preference?",
            "year": "Are you leaning toward something recent, or more of a timeless classic?",
            "reference_movie": "Is there a movie you recently loved and want a similar vibe to? If not, no worries."
        }
        return questions.get(field, "Great, I have a good direction. Want me to pick a movie for you?")


def build_recommendation_query(state):
    parts = []
    if state.get("genre"):
        parts.append(str(state["genre"]))
    if state.get("year"):
        year = extract_year(str(state["year"]))
        parts.append(f"from {year}" if year else str(state["year"]))
    elif state.get("era") == "recent":
        parts.append("from 2020")
    elif state.get("era") == "classic":
        parts.append("classic older movie")
    if state.get("platform"):
        parts.append(str(state["platform"]))
    if state.get("occasion"):
        parts.append(str(state["occasion"]))
    if state.get("reference_movie"):
        parts.append(f"similar to {state['reference_movie']}")
    return " ".join(parts).strip()


def recommend_from_state(state, user_text):
    query = state.get("last_query") or build_recommendation_query(state)
    state["last_query"] = query
    reference = state.get("reference_movie")
    state.setdefault("shown_titles", [])
    shown = state.get("shown_titles", [])

    if reference:
        movie_title = find_movie_title(reference) or reference
        result = handle_similar(movie_title, query, top_n=1, exclude_titles=shown)
        if not result.get("results"):
            result = handle_search(query, top_n=1, exclude_titles=shown)
    else:
        result = handle_search(query, top_n=1, exclude_titles=shown)

    if result.get("results"):
        title = result["results"][0]["title"]
        if title not in state["shown_titles"]:
            state["shown_titles"].append(title)

    state["done"] = True
    result["interview"] = state.copy()
    return result


def call_openai(user_text, result):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return fallback_reply(user_text, result)

    try:
        import urllib.request
        language = "Hebrew" if is_hebrew(user_text) else "English"
        intent = result.get("intent", "search")
        results = result.get("results", [])
        clusters = result.get("clusters", [])
        data_block = ""

        if results:
            for r in results[:1]:
                streaming = r.get("streaming") or "Not available in streaming dataset"
                data_block += (
                    f"- {r['title']} ({r['year']}), genres: {r['genres']}, "
                    f"rating: {r['rating']}/10, streaming: {streaming}, score: {r['score']}\n"
                )
        elif clusters:
            for c in clusters:
                data_block += (
                    f"- Cluster {c['cluster']}: {c['name']}, count: {c['count']}, "
                    f"top genres: {c['top_genres']}, avg rating: {c['avg_rating']}\n"
                )
        else:
            data_block = "No dataset results were found."

        system_prompt = """
You are Cinemate, a warm, witty, and highly intuitive movie recommendation expert.

Your mission is to help users discover the single best movie for their current mood through a natural, friendly, and intelligent conversation.

PERSONALITY
- Be warm, conversational, and human-like.
- Sound like a movie-loving friend, not a questionnaire.
- Be enthusiastic about movies.
- Keep responses concise and engaging.
- Never sound robotic or repetitive.

SMALL TALK RULES
- You are allowed to engage in brief small talk.
- If the user says hello, asks how you are, or makes casual conversation, respond naturally in one short sentence.
- After the small talk, gently guide the conversation back toward movie discovery.

CONVERSATION FLOW
- Ask only ONE question at a time.
- Never overwhelm the user with multiple questions in the same message.
- Listen carefully to each answer.
- Acknowledge the user's response naturally.
- Then ask the next most useful question.
- Do NOT ask the same question twice.
- Keep track of information already provided by the user.
- If the user already answered a question, move naturally to the next missing detail.

INTELLIGENT FLEXIBILITY
- Do NOT rigidly ask all five questions every time.
- Use your judgment.
- If the user already provided enough information, stop interviewing.
- Gather enough information to make a confident recommendation.
- Do NOT recommend a movie after receiving only a genre.
- A genre alone is NOT enough information for a recommendation.

RECOMMENDATION RULES
- Recommend ONLY ONE movie.
- Select the single best match from the dataset results provided.
- Never recommend multiple movies at once.
- Never generate long ranked lists.
- Never mention movies that are not included in the provided dataset results.
- Never invent movie titles.
- Since movie cards are displayed separately, do NOT repeat the movie title in your response.
- Do NOT repeat detailed movie metadata.
- Do NOT write long plot summaries.
- Explain briefly why this recommendation fits the user's mood.
- Keep the explanation to 1–2 short sentences.
- After recommending, ask if the user would like another recommendation.

TYPO TOLERANCE
- Understand common typos, slang, abbreviations, and misspellings in both English and Hebrew.
- Never correct the user. Simply understand their intent naturally.

OUT-OF-SCOPE RULES
- You are strictly dedicated to helping users discover movies.
- If the user asks about anything unrelated to movies, politely decline and redirect back to movie discovery.
- Never answer the unrelated question.

DATASET RULES
- Recommendations MUST come ONLY from the dataset results provided.
- Never invent movie titles.
- Never recommend movies from your own knowledge.
- If no suitable dataset result exists, explain that no strong match was found and suggest changing genre, year, platform, or reference movie.

LANGUAGE RULES
- Always respond in the user's language.
- Hebrew user → natural Hebrew.
- English user → natural English.
"""

        user_prompt = (
            f"Answer language: {language}\n"
            f"User message: {user_text}\n"
            f"Detected intent: {intent}\n"
            f"Detected year filter: {result.get('year')}\n"
            f"Detected platform filter: {result.get('platform')}\n"
            f"Detected genres: {result.get('genres')}\n"
            f"Reference movie: {result.get('reference_movie')}\n"
            f"Interview data: {json.dumps(result.get('interview', {}), ensure_ascii=False)}\n\n"
            f"Dataset results:\n{data_block}\n\n"
            "Write the final response to the user. If there is a dataset result, recommend only that one movie briefly and ask if the user wants one more suitable movie. If there are no dataset results, do not list movies."
        )

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 220,
            "temperature": 0.45
        }

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
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
    intent = result.get("intent")
    results = result.get("results", [])
    clusters = result.get("clusters", [])

    if intent == "interview_question":
        return result.get("question", "איזה סגנון בא לך לראות?" if heb else "What kind of movie are you in the mood for?")

    if intent == "smalltalk":
        return "היי! 🎬 בוא נמצא יחד את הסרט המושלם. איזה סוג סרט בא לך לראות?" if heb else "Hi! 🎬 Let’s find the perfect movie together. What kind of movie are you in the mood for?"

    if intent == "out_of_scope":
        return friendly_out_of_scope_reply(user_text)

    if clusters:
        return "מצאתי את קבוצות הסרטים המרכזיות במאגר." if heb else "I found the main movie clusters in the dataset."

    if not results:
        return "לא מצאתי התאמה טובה בדאטה. אפשר לנסות לשנות שם סרט, שנה, ז׳אנר או פלטפורמה." if heb else "I couldn’t find a good match in the dataset. Try changing the movie name, year, genre, or platform."

    if result.get("exhausted"):
        return "אין לי עוד המלצה טובה לפי אותם פרטים. אפשר להתחיל חיפוש חדש." if heb else "I don’t have another strong match for the same details. You can start a new search."

    return "נראה לי שמצאתי התאמה ממש טובה למה שחיפשת 🎬 רוצה שאציע עוד סרט מתאים?" if heb else "I think this is a really good match for what you described 🎬 Would you like one more suitable movie?"


# ============================================================== 
# 9. ROUTES
# ============================================================== 

@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "movies": int(len(df))})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_text = normalize_user_text(data.get("message", ""))
    key = get_conversation_key()
    heb = is_hebrew(user_text) if user_text else True

    if key not in CONVERSATIONS:
        CONVERSATIONS[key] = new_state()

    state = CONVERSATIONS[key]

    # Explicit restart: reset and start a fresh flexible interview.
    if wants_reset(user_text):
        CONVERSATIONS[key] = new_state()
        state = CONVERSATIONS[key]
        q = next_interview_question(state, heb)
        return jsonify({"intent": "interview_question", "reply": q, "question": q, "results": [], "reset": True})

    # Empty message / page ping: do not consume an interview answer.
    if not user_text:
        q = next_interview_question(state, heb) if not state.get("done") else (
            "היי 🎬 רוצה שנתחיל חיפוש חדש או שאציע עוד סרט לפי הבחירות הקודמות?" if heb else
            "Hi 🎬 Would you like to start a new search or get one more movie based on your previous choices?"
        )
        return jsonify({"intent": "interview_question", "reply": q, "question": q, "results": []})

    # Friendly small talk is allowed. It should NOT be saved as genre/year/platform.
    if is_smalltalk(user_text):
        reply = smalltalk_reply(user_text, state) if not state.get("done") else (
            "הכול טוב 🙂 רוצה שאציע עוד סרט לפי הבחירות שלך, או שנתחיל חיפוש חדש?"
            if heb else
            "All good 🙂 Would you like one more movie based on your choices, or should we start a new search?"
        )
        return jsonify({"intent": "smalltalk", "reply": reply, "results": []})

    # Off-topic guardrail.
    if maybe_out_of_scope_during_interview(user_text):
        return jsonify({"intent": "out_of_scope", "reply": friendly_out_of_scope_reply(user_text), "results": []})

    # After a recommendation, "yes / more" means one additional movie from the same query.
    if state.get("done") and wants_more(user_text):
        result = recommend_from_state(state, user_text)
        CONVERSATIONS[key] = state
        result["reply"] = call_openai(user_text, result)
        return jsonify(result)

    # If the user says no after a recommendation, finish politely without forcing more questions.
    if state.get("done") and is_none_answer(user_text):
        reply = "סבבה 🎬 כשתרצה סרט חדש פשוט כתוב 'סרט חדש'." if heb else "No problem 🎬 When you want a new movie, just type 'new movie'."
        return jsonify({"intent": "smalltalk", "reply": reply, "results": []})

    # If a completed conversation receives a new movie preference, start a new flexible interview.
    if state.get("done"):
        state = new_state()
        CONVERSATIONS[key] = state

    # Flexible interview mode:
    # Extract any details the user already gave, then either ask the best missing question or recommend.
    state = merge_preferences_from_text(state, user_text)
    CONVERSATIONS[key] = state

    if has_enough_for_recommendation(state):
        result = recommend_from_state(state, user_text)
        CONVERSATIONS[key] = state
        result["reply"] = call_openai(user_text, result)
        return jsonify(result)

    q = next_interview_question(state, heb)
    return jsonify({"intent": "interview_question", "reply": q, "question": q, "results": []})


# ==============================================================

# 10. HTML - compact cinema style

# ==============================================================



HTML_PAGE = f"""<!DOCTYPE html>

<html lang="he" dir="rtl">

<head>

<meta charset="UTF-8">

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>צ׳אטבוט סרטים</title>

<link href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700;900&display=swap" rel="stylesheet">

<style>

:root {{

  --bg:#080808;

  --red:#d71920;

  --red2:#ff3040;

  --gold:#ffd166;

  --cream:#fff7ec;

  --text:#f6f1ea;

  --dark:#141414;

  --muted:#b7b0aa;

}}

* {{ box-sizing:border-box; }}

body {{

  margin:0;

  font-family:'Heebo', sans-serif;

  direction:rtl;

  color:var(--text);

  min-height:100vh;

  background:

    linear-gradient(rgba(0,0,0,.76), rgba(0,0,0,.72)),

    radial-gradient(circle at 20% 10%, rgba(255,48,64,.28), transparent 25%),

    radial-gradient(circle at 80% 20%, rgba(255,209,102,.16), transparent 24%),

    #080808;

  overflow-x:hidden;

}}

body::before {{

  content:"";

  position:fixed;

  inset:0;

  pointer-events:none;

  background:

    repeating-linear-gradient(90deg, rgba(255,255,255,.03) 0 2px, transparent 2px 84px),

    linear-gradient(90deg, transparent, rgba(255,255,255,.09), transparent);

  animation:spot 8s linear infinite;

  opacity:.7;

}}

@keyframes spot {{

  from {{ background-position:-500px 0, -900px 0; }}

  to {{ background-position:500px 0, 900px 0; }}

}}

.marquee {{

  position:fixed;

  top:0;

  left:0;

  right:0;

  height:10px;

  background:repeating-linear-gradient(90deg, var(--gold) 0 18px, #5b0004 18px 36px);

  box-shadow:0 0 18px rgba(255,209,102,.7);

  z-index:3;

}}

header {{

  position:relative;

  z-index:4;

  padding:18px 42px 12px;

  display:flex;

  align-items:center;

  justify-content:space-between;

}}

.logo {{

  font-size:30px;

  font-weight:900;

  letter-spacing:.5px;

}}

.logo span {{ color:var(--red2); }}

.badge {{

  background:rgba(255,255,255,.08);

  border:1px solid rgba(255,255,255,.14);

  padding:8px 16px;

  border-radius:999px;

  color:var(--gold);

  font-weight:700;

  font-size:15px;

}}

.hero {{

  position:relative;

  z-index:2;

  text-align:center;

  padding:8px 18px 12px;

}}

.hero h1 {{

  margin:10px 0 6px;

  font-size:clamp(38px, 7vw, 74px);

  line-height:1;

  font-weight:900;

  text-shadow:0 6px 0 rgba(215,25,32,.45), 0 0 28px rgba(255,48,64,.24);

}}

.hero p {{

  margin:0 auto;

  color:#ddd6d0;

  font-size:clamp(18px, 2.5vw, 26px);

}}

.stage {{

  position:relative;

  z-index:2;

  width:min(1120px, 92vw);

  margin:18px auto 34px;

  background:rgba(14,14,14,.88);

  border:1px solid rgba(255,255,255,.13);

  border-radius:28px;

  box-shadow:0 24px 80px rgba(0,0,0,.55), inset 0 0 0 1px rgba(255,255,255,.04);

  overflow:hidden;

}}

.stage-top {{

  height:42px;

  background:linear-gradient(90deg, #260003, #9e1018, #260003);

  display:flex;

  align-items:center;

  justify-content:center;

  color:var(--gold);

  font-weight:900;

  letter-spacing:2px;

}}

.content {{ padding:24px; }}

.quick-title {{ font-size:18px; font-weight:900; margin-bottom:10px; color:var(--cream); }}

.chips {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:16px; }}

.chip {{

  border:1px solid rgba(255,209,102,.34);

  color:var(--cream);

  background:rgba(255,209,102,.08);

  padding:9px 14px;

  border-radius:999px;

  cursor:pointer;

  transition:.18s;

  font-size:15px;

}}

.chip:hover {{ background:rgba(215,25,32,.45); transform:translateY(-2px); }}

.chat {{

  background:rgba(255,247,236,.96);

  color:#222;

  border-radius:22px;

  height:420px;

  overflow-y:auto;

  padding:20px;

  border:5px solid rgba(215,25,32,.18);

}}

.msg {{ display:flex; margin:12px 0; }}

.msg.user {{ justify-content:flex-start; }}

.msg.bot {{ justify-content:flex-end; }}

.msg.bot.has-cards {{ flex-direction:column; align-items:flex-end; }}

.bubble {{

  max-width:76%;

  padding:13px 16px;

  border-radius:20px;

  line-height:1.65;

  font-size:16px;

  box-shadow:0 6px 16px rgba(0,0,0,.08);

  white-space:pre-line;

}}

.user .bubble {{ background:linear-gradient(135deg, var(--red), var(--red2)); color:#fff; border-bottom-left-radius:4px; }}

.bot .bubble {{ background:#f2f2f2; color:#222; border-bottom-right-radius:4px; }}

.cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:12px; margin-top:10px; width:100%; }}

.card {{ background:white; border:1px solid #eee; border-radius:16px; padding:14px; color:#222; box-shadow:0 8px 20px rgba(0,0,0,.08); }}

.card-title {{ font-weight:900; color:#b10e15; font-size:17px; }}

.meta {{ color:#555; font-size:13px; margin:5px 0; }}

.genres {{ font-size:13px; color:#7a4b00; font-weight:700; margin-bottom:5px; }}

.desc {{ color:#444; font-size:13px; line-height:1.45; }}

.stream {{ margin-top:6px; color:#111; font-size:13px; font-weight:800; }}

.input-row {{ display:flex; gap:10px; margin-top:14px; }}

#inp {{

  flex:1;

  border:none;

  outline:none;

  border-radius:18px;

  padding:15px 18px;

  font-family:'Heebo', sans-serif;

  font-size:17px;

  background:#fff;

}}

#btn {{

  border:none;

  border-radius:18px;

  padding:0 24px;

  background:linear-gradient(135deg, var(--red), #760006);

  color:#fff;

  font-size:18px;

  font-weight:900;

  cursor:pointer;

  box-shadow:0 10px 22px rgba(215,25,32,.35);

}}

.typing {{ display:inline-flex; gap:5px; align-items:center; }}

.dot {{ width:7px; height:7px; background:#b10e15; border-radius:50%; animation:bounce 1s infinite; }}

.dot:nth-child(2){{animation-delay:.2s}}

.dot:nth-child(3){{animation-delay:.4s}}

@keyframes bounce {{

  0%,80%,100%{{transform:translateY(0); opacity:.4}}

  40%{{transform:translateY(-6px); opacity:1}}

}}

@media(max-width:720px){{

  header {{ padding:16px 18px 8px; }}

  .badge {{ display:none; }}

  .stage {{ width:94vw; }}

  .content {{ padding:15px; }}

  .chat {{ height:390px; }}

  .bubble,.cards {{ max-width:94%; }}

  .input-row {{ flex-direction:column; }}

  #btn {{ padding:13px; }}

}}

</style>

</head>

<body>

<div class="marquee"></div>

<header>

  <div class="logo">🎬 <span>Cinemate</span></div>

</header>



<section class="hero">

  <h1>מחפשים את הסרט המושלם? 🍿</h1>

</section>



<main class="stage">

  <div class="stage-top">NOW SHOWING • MOVIE AGENT • NOW SHOWING</div>

  <div class="content">

<div id="chat" class="chat">

      <div class="msg bot">

        <div class="bubble">היי, ברוכים הבאים ל-Cinemate 🎬 בואו נמצא יחד סרט שמתאים בדיוק למצב הרוח שלכם. נתחיל בקטנה: איזה סגנון בא לכם לראות?</div>

      </div>

    </div>



    <div class="input-row">

      <input id="inp" placeholder="ענו כאן לשאלה של Cinemate..." autocomplete="off">

      <button id="btn">שליחה</button>

    </div>

  </div>

</main>



<script>

const chat = document.getElementById('chat');

const inp = document.getElementById('inp');

const btn = document.getElementById('btn');
const sessionId = crypto.randomUUID();
localStorage.setItem('cinemate_session', sessionId);



function esc(s){{

  return String(s || '').replace(/[&<>"']/g, m => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m]));

}}



function add(role, html){{

  const d = document.createElement('div');

  d.className = 'msg ' + role;

  d.innerHTML = html;

  chat.appendChild(d);

  chat.scrollTop = chat.scrollHeight;

}}



function addTyping(){{

  add('bot', '<div class="bubble" id="typing"><span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span></div>');

}}



function rmTyping(){{

  const t = document.getElementById('typing');

  if(t) t.parentElement.remove();

}}



function cards(results){{

  if(!results || !results.length) return '';

  let h = '<div class="cards">';

  results.slice(0, 1).forEach(r => {{

    h += `<div class="card">

      <div class="card-title">${{esc(r.rank)}}. ${{esc(r.title)}}</div>

      <div class="meta">${{esc(r.year)}} • ⭐ ${{esc(r.rating)}}/10 • התאמה ${{esc(r.score)}}</div>

      <div class="genres">${{esc(r.genres)}}</div>

      ${{r.streaming ? `<div class="stream">זמין ב: ${{esc(r.streaming)}}</div>` : ''}}


    </div>`;

  }});

  h += '</div>';

  return h;

}}



function clusters(list){{

  if(!list) return '';

  let h = '<div class="cards">';

  list.forEach(c => {{

    h += `<div class="card">

      <div class="card-title">${{esc(c.name)}}</div>

      <div class="meta">${{esc(c.count)}} סרטים • ממוצע ${{esc(c.avg_rating)}}</div>

      <div class="genres">${{esc(c.top_genres)}}</div>

    </div>`;

  }});

  return h + '</div>';

}}



function send(){{

  const text = inp.value.trim();

  if(!text) return;

  add('user', '<div class="bubble">' + esc(text) + '</div>');

  inp.value = '';

  addTyping();



  fetch('/chat', {{

    method:'POST',

    headers:{{'Content-Type':'application/json', 'X-Session-Id': sessionId}},

    body:JSON.stringify({{message:text}})

  }})

  .then(r => r.json())

  .then(data => {{

    rmTyping();

    if (data.reset) {{
      chat.innerHTML = '';
    }}

    const hasResults = data.results && data.results.length > 0;

    const hasCluster = data.intent === 'cluster_info' && data.clusters && data.clusters.length > 0;

    // Strip markdown bold/italic that GPT sometimes adds

    const cleanReply = (data.reply || '').replace(/\*\*([^*]+)\*\*/g, '$1').replace(/\*([^*]+)\*/g, '$1');

    let extra = '';

    if (hasCluster) {{

      extra = clusters(data.clusters);

    }} else if (hasResults) {{

      extra = cards(data.results);

    }}

    const msgClass = (hasResults || hasCluster) ? 'bot has-cards' : 'bot';

    add(msgClass, '<div class="bubble">' + esc(cleanReply) + '</div>' + extra);

  }})

  .catch(() => {{

    rmTyping();

    add('bot', '<div class="bubble">משהו השתבש. נסו שוב בעוד רגע.</div>');

  }});

}}



function go(q){{

  inp.value = q;

  send();

}}



btn.onclick = send;

inp.addEventListener('keydown', e => {{

  if(e.key === 'Enter') send();

}});

</script>

</body>

</html>"""



# ==============================================================

# 11. RUN

# ==============================================================



if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port, debug=False)


