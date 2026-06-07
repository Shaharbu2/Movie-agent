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
# 1. LOAD & PREPARE DATA - optimized for Render free memory
# ==============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "movies_master.csv")

NEEDED_COLUMNS = [
    "title", "overview", "genres", "keywords", "available_on",
    "vote_average", "popularity", "runtime", "vote_count", "release_year",
    "Netflix", "Hulu", "Prime Video", "Disney+"
]

# Read only columns that exist, max 50,000 rows
header_cols = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
usecols = [c for c in NEEDED_COLUMNS if c in header_cols]
df = pd.read_csv(DATA_PATH, usecols=usecols, nrows=50000)

# Ensure all expected columns exist
for col in NEEDED_COLUMNS:
    if col not in df.columns:
        if col in ["Netflix", "Hulu", "Prime Video", "Disney+"]:
            df[col] = 0
        elif col in ["vote_average", "popularity", "runtime", "vote_count", "release_year"]:
            df[col] = 0
        elif col == "available_on":
            df[col] = "לא זמין בסטרימינג"
        else:
            df[col] = ""

df["title"] = df["title"].fillna("").astype(str)
df["overview"] = df["overview"].fillna("").astype(str)
df["genres"] = df["genres"].fillna("").astype(str)
df["keywords"] = df["keywords"].fillna("").astype(str)
df["available_on"] = df["available_on"].fillna("לא זמין בסטרימינג").astype(str)

for c in ["vote_average", "popularity", "runtime", "vote_count", "release_year"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

for col in ["Netflix", "Hulu", "Prime Video", "Disney+"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int8)

# Keep memory lower
for c in ["vote_average", "popularity", "runtime", "vote_count"]:
    df[c] = df[c].astype(np.float32)
df["release_year"] = df["release_year"].astype(np.int16)

# ==============================================================
# 2. HELPERS
# ==============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
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

# ==============================================================
# 3. CLUSTERING - sparse
# ==============================================================

numeric_features = ["vote_average", "popularity", "runtime", "vote_count"]
scaler = MinMaxScaler()
numeric_scaled = scaler.fit_transform(df[numeric_features]).astype(np.float32)
numeric_sparse = csr_matrix(numeric_scaled)

genres_vec = vectorize_column("genres", max_features=35)
keywords_vec = vectorize_column("keywords", max_features=35)

cluster_data = hstack([numeric_sparse, genres_vec, keywords_vec], format="csr")

kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, n_init=2, batch_size=2048)
df["cluster"] = kmeans.fit_predict(cluster_data).astype(np.int8)

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

# Do not keep another full overview_clean column in df
overview_clean_list = [clean_text(x) for x in df["overview"].tolist()]
tfidf = TfidfVectorizer(stop_words="english", max_features=1500, ngram_range=(1, 2), dtype=np.float32)
tfidf_matrix = tfidf.fit_transform(overview_clean_list)

# Similarity features for "similar to"
sim_data_sparse = hstack([numeric_sparse, genres_vec, keywords_vec], format="csr")

# ==============================================================
# 5. ANOMALY DETECTION - light
# ==============================================================

iso_features = ["popularity", "vote_average", "vote_count", "runtime"]
iso_scaler = MinMaxScaler()
iso_scaled = iso_scaler.fit_transform(df[iso_features]).astype(np.float32)
iso = IsolationForest(n_estimators=15, max_samples=4096, contamination=0.05, random_state=42)
df["anomaly"] = iso.fit_predict(iso_scaled).astype(np.int8)
df["anomaly_score"] = iso.decision_function(iso_scaled).astype(np.float32)

# ==============================================================
# 6. INTENT + FILTER DETECTION
# ==============================================================

GENRE_KEYWORD_MAP = {
    "action": "Action", "אקשן": "Action", "fight": "Action", "battle": "Action",
    "adventure": "Adventure", "הרפתקה": "Adventure",
    "animation": "Animation", "אנימציה": "Animation", "cartoon": "Animation",
    "comedy": "Comedy", "קומדיה": "Comedy", "funny": "Comedy", "מצחיק": "Comedy",
    "crime": "Crime", "פשע": "Crime", "detective": "Crime",
    "documentary": "Documentary", "דוקומנטרי": "Documentary",
    "drama": "Drama", "דרמה": "Drama", "emotional": "Drama",
    "family": "Family", "משפחה": "Family", "kids": "Family", "ילדים": "Family",
    "fantasy": "Fantasy", "פנטזיה": "Fantasy", "magic": "Fantasy", "קסם": "Fantasy",
    "history": "History", "היסטוריה": "History", "war": "War", "מלחמה": "War",
    "horror": "Horror", "אימה": "Horror", "scary": "Horror", "מפחיד": "Horror",
    "mystery": "Mystery", "מסתורין": "Mystery",
    "romance": "Romance", "רומנטי": "Romance", "romantic": "Romance", "love": "Romance", "אהבה": "Romance",
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
    # catches 1900-2099
    m = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    return int(m.group(1)) if m else None

def extract_platform(text):
    t = text.lower()
    for platform, pats in PLATFORM_PATTERNS.items():
        if any(p in t for p in pats):
            return platform
    return None

MOVIE_WORDS = [
    "סרט", "סרטים", "קולנוע", "נטפליקס", "דיסני", "פריים", "הולו",
    "קומדיה", "אקשן", "אימה", "דרמה", "רומנטי", "רומנטיקה", "מתח",
    "אנימציה", "ילדים", "פנטזיה", "מדע בדיוני", "דומה", "כמו",
    "movie", "movies", "film", "films", "cinema", "netflix", "hulu",
    "prime", "disney", "comedy", "action", "horror", "drama", "romance",
    "thriller", "animation", "similar", "like", "recommend"
]

def is_movie_related(text):
    t = text.lower().strip()
    if extract_year(t) or extract_platform(t) or extract_genres(t):
        return True
    return any(w in t for w in MOVIE_WORDS)

def no_relevant_answer():
    return {
        "intent": "search",
        "reply": "לא מצאתי התאמה מספיק טובה לשאלה שלך. אני צ׳אטבוט סרטים 🎬 אפשר לשאול אותי למשל: סרטי קומדיה משנת 2000, סרטים בנטפליקס, סרטים דומים ל-Inception או סרטים עם דירוג גבוה.",
        "results": []
    }

def apply_filters(base_df, text):
    filtered = base_df
    year = extract_year(text)
    platform = extract_platform(text)

    if year is not None:
        filtered = filtered[filtered["release_year"] == year]

    if platform is not None and platform in filtered.columns:
        filtered = filtered[filtered[platform] == 1]

    return filtered, year, platform

def detect_intent(text):
    t = text.lower()
    for pat in [r"similar to (.+)", r"like (.+)", r"movies like (.+)", r"דומה ל(.+)", r"כמו (.+)", r"סרטים כמו (.+)"]:
        m = re.search(pat, t)
        if m:
            return "similar", m.group(1).strip().rstrip("?.,!")
    if any(k in t for k in ["unusual", "anomaly", "flop", "hidden gem", "underrated", "פלופ", "יהלום נסתר", "חריג", "מוזר"]):
        return "anomaly", text
    if any(k in t for k in ["cluster", "קלסטר", "קבוצה", "סוגי סרטים"]):
        return "cluster_info", text
    return "search", text

def get_streaming(row):
    platforms = []
    if row.get("Netflix", 0) == 1: platforms.append("Netflix")
    if row.get("Hulu", 0) == 1: platforms.append("Hulu")
    if row.get("Prime Video", 0) == 1: platforms.append("Prime")
    if row.get("Disney+", 0) == 1: platforms.append("Disney+")
    return ", ".join(platforms) if platforms else ""

def row_to_result(rank, idx, score=0):
    row = df.iloc[int(idx)]
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

# ==============================================================
# 7. HANDLERS - max 3 results
# ==============================================================

def handle_search(user_text, top_n=3):
    if not is_movie_related(user_text):
        return no_relevant_answer()

    filtered, year, platform = apply_filters(df, user_text)

    if filtered.empty:
        parts = []
        if year: parts.append(f"משנת {year}")
        if platform: parts.append(f"שזמינים ב-{platform}")
        msg = "לא מצאתי סרטים " + " ".join(parts) + ". נסי לשנות שנה, ז'אנר או פלטפורמה."
        return {"intent": "search", "reply": msg, "results": [], "year": year, "platform": platform}

    matched_genres = extract_genres(user_text)
    matched_clusters = genres_to_clusters(matched_genres)
    cleaned = clean_text(user_text)

    # If the message has no meaningful searchable text and no filters, don't invent results
    if not cleaned and not matched_genres and not year and not platform:
        return no_relevant_answer()

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

    # Important: do not return random top 3 when scores are too weak.
    # With only a year/platform filter, allow ranked results by popularity/rating.
    has_clear_filter = bool(year or platform or matched_genres)
    best_score = float(combined.max()) if len(combined) else 0.0

    if not has_clear_filter and best_score < 0.08:
        return no_relevant_answer()

    if has_clear_filter and best_score == 0:
        # For requests like "סרטים משנת 2000" where text similarity is low,
        # sort filtered movies by rating and vote_count instead of returning unrelated matches.
        filtered_ranked = filtered.sort_values(["vote_average", "vote_count", "popularity"], ascending=False).head(top_n)
        results = []
        for i, (idx, row) in enumerate(filtered_ranked.iterrows()):
            results.append(row_to_result(i + 1, idx, row["vote_average"] / 10))
    else:
        order = combined.argsort()[-top_n:][::-1]
        # Keep only reasonably relevant results
        chosen = [pos for pos in order if combined[pos] >= max(0.04, best_score * 0.25)]
        if not chosen:
            return no_relevant_answer()
        results = [row_to_result(i + 1, indices[pos], combined[pos]) for i, pos in enumerate(chosen[:top_n])]

    reply = "מצאתי עד 3 סרטים שמתאימים לבקשה שלך"
    extras = []
    if year: extras.append(f"שנת {year}")
    if platform: extras.append(f"זמינים ב-{platform}")
    if matched_genres: extras.append("ז'אנרים: " + ", ".join(matched_genres))
    if extras:
        reply += " (" + " | ".join(extras) + ")"
    reply += ":"
    return {"intent": "search", "reply": reply, "results": results, "genres": matched_genres, "year": year, "platform": platform}

def handle_similar(movie_title, top_n=3):
    matches = df[df["title"].str.lower().str.contains(movie_title.lower(), na=False)]
    if matches.empty:
        return {"intent": "similar", "reply": f"לא מצאתי את הסרט '{movie_title}' במסד הנתונים.", "results": []}

    idx = int(matches.index[0])
    found = df.loc[idx, "title"]
    scores = cosine_similarity(sim_data_sparse[idx], sim_data_sparse).flatten()
    scores[idx] = -1
    top_idx = scores.argsort()[-top_n:][::-1]

    results = [row_to_result(i + 1, tidx, scores[tidx]) for i, tidx in enumerate(top_idx)]
    return {"intent": "similar", "reply": f"הנה עד 3 סרטים דומים ל-{found}:", "results": results}

def handle_anomaly(user_text, top_n=3):
    filtered, year, platform = apply_filters(df, user_text)
    anomalies = filtered[filtered["anomaly"] == -1].copy()

    if anomalies.empty:
        return {"intent": "anomaly", "reply": "לא מצאתי חריגות שמתאימות לסינון שביקשת.", "results": []}

    t = user_text.lower()
    if any(k in t for k in ["flop", "פלופ", "budget", "תקציב"]):
        subset = anomalies[(anomalies["vote_average"] < 5.5) & (anomalies["popularity"] > anomalies["popularity"].median())]
        label = "פלופים — פופולריים אבל עם ציון נמוך"
    elif any(k in t for k in ["hidden gem", "יהלום נסתר", "underrated", "מוערך בחסר"]):
        subset = anomalies[(anomalies["vote_average"] >= 7.5) & (anomalies["vote_count"] < anomalies["vote_count"].quantile(0.45))]
        label = "יהלומים נסתרים — ציון גבוה ומעט חשיפה"
    elif any(k in t for k in ["long", "short", "ארוך", "קצר", "runtime"]):
        subset = anomalies[(anomalies["runtime"] > 180) | (anomalies["runtime"] < 60)]
        label = "סרטים עם משך זמן חריג"
    else:
        subset = anomalies.sort_values("anomaly_score").head(top_n)
        label = "חריגות מעניינות במסד הנתונים"

    if subset.empty:
        subset = anomalies.sort_values("anomaly_score").head(top_n)

    subset = subset.head(top_n)
    results = []
    for i, (idx, row) in enumerate(subset.iterrows()):
        results.append(row_to_result(i + 1, idx, row["anomaly_score"]))
    return {"intent": "anomaly", "reply": f"הנה עד 3 {label}:", "results": results}

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
    return {"intent": "cluster_info", "reply": "הנה פירוט קבוצות הסרטים:", "clusters": summary}

# ==============================================================
# 8. OPTIONAL OPENAI
# ==============================================================

def call_openai(user_text, results, intent):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or not results:
        return None
    try:
        import urllib.request
        movies = ""
        for r in results[:3]:
            streaming = (" | זמין ב: " + r["streaming"]) if r.get("streaming") else ""
            movies += f"- {r['title']} ({r['year']}): {r['genres']}, {r['rating']}/10{streaming}\n"
        prompt = (
            "ענה בעברית בלבד, קצר וידידותי. המשתמש שאל: "
            + user_text
            + ". הסרטים שנמצאו:\n"
            + movies
            + "כתוב 2-3 משפטים בלבד. ציין את שמות הסרטים והסבר למה הם מתאימים."
        )
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 180,
            "temperature": 0.7
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
        return None

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
    user_text = data.get("message", "").strip()
    if not user_text:
        return jsonify({"reply": "תכתבי לי איזה סרט או מצב רוח את מחפשת 🎬", "results": []})

    intent, payload = detect_intent(user_text)

    if intent == "similar":
        result = handle_similar(payload)
    elif intent == "anomaly":
        result = handle_anomaly(user_text)
    elif intent == "cluster_info":
        result = handle_cluster_info(user_text)
    else:
        result = handle_search(user_text)

    if intent != "cluster_info":
        ai_reply = call_openai(user_text, result.get("results", []), intent)
        if ai_reply:
            result["ai_reply"] = ai_reply

    return jsonify(result)

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
.content {{
  padding:24px;
}}
.quick-title {{
  font-size:18px;
  font-weight:900;
  margin-bottom:10px;
  color:var(--cream);
}}
.chips {{
  display:flex;
  flex-wrap:wrap;
  gap:10px;
  margin-bottom:16px;
}}
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
.chip:hover {{
  background:rgba(215,25,32,.45);
  transform:translateY(-2px);
}}
.chat {{
  background:rgba(255,247,236,.96);
  color:#222;
  border-radius:22px;
  height:420px;
  overflow-y:auto;
  padding:20px;
  border:5px solid rgba(215,25,32,.18);
}}
.msg {{
  display:flex;
  margin:12px 0;
}}
.msg.user {{ justify-content:flex-start; }}
.msg.bot {{ justify-content:flex-end; }}
.bubble {{
  max-width:76%;
  padding:13px 16px;
  border-radius:20px;
  line-height:1.65;
  font-size:16px;
  box-shadow:0 6px 16px rgba(0,0,0,.08);
}}
.user .bubble {{
  background:linear-gradient(135deg, var(--red), var(--red2));
  color:#fff;
  border-bottom-left-radius:4px;
}}
.bot .bubble {{
  background:#f2f2f2;
  color:#222;
  border-bottom-right-radius:4px;
}}
.ai-box {{
  margin-top:8px;
  max-width:76%;
  background:#fff8df;
  border:1px solid #f1cf69;
  color:#40320b;
  border-radius:18px;
  padding:12px 15px;
  line-height:1.6;
}}
.cards {{
  display:grid;
  grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));
  gap:12px;
  margin-top:10px;
  max-width:86%;
}}
.card {{
  background:white;
  border:1px solid #eee;
  border-radius:16px;
  padding:14px;
  color:#222;
  box-shadow:0 8px 20px rgba(0,0,0,.08);
}}
.card-title {{
  font-weight:900;
  color:#b10e15;
  font-size:17px;
}}
.meta {{
  color:#555;
  font-size:13px;
  margin:5px 0;
}}
.genres {{
  font-size:13px;
  color:#7a4b00;
  font-weight:700;
  margin-bottom:5px;
}}
.desc {{
  color:#444;
  font-size:13px;
  line-height:1.45;
}}
.stream {{
  margin-top:6px;
  color:#111;
  font-size:13px;
  font-weight:800;
}}
.input-row {{
  display:flex;
  gap:10px;
  margin-top:14px;
}}
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
.typing {{
  display:inline-flex;
  gap:5px;
  align-items:center;
}}
.dot {{
  width:7px;
  height:7px;
  background:#b10e15;
  border-radius:50%;
  animation:bounce 1s infinite;
}}
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
  .bubble,.ai-box,.cards {{ max-width:94%; }}
  .input-row {{ flex-direction:column; }}
  #btn {{ padding:13px; }}
}}
</style>
</head>
<body>
<div class="marquee"></div>
<header>
  <div class="logo">🎬 צ׳אטבוט <span>סרטים</span></div>
  <div class="badge">מבוסס AI + {len(df):,} סרטים</div>
</header>

<section class="hero">
  <h1>מחפשים את הסרט המושלם? 🍿</h1>
  <p>שאלו על שנה, ז׳אנר, נטפליקס, סרטים דומים או חריגות מעניינות בדאטה</p>
</section>

<main class="stage">
  <div class="stage-top">NOW SHOWING • MOVIE AGENT • NOW SHOWING</div>
  <div class="content">
    <div class="quick-title">דוגמאות לשאלות:</div>
    <div class="chips">
      <button class="chip" onclick="go('תמצא לי סרט קומדיה משנת 2000')">קומדיה משנת 2000</button>
      <button class="chip" onclick="go('סרטי אקשן שקיימים בנטפליקס')">אקשן בנטפליקס</button>
      <button class="chip" onclick="go('movies similar to Inception')">דומה ל-Inception</button>
      <button class="chip" onclick="go('find me hidden gems')">יהלומים נסתרים</button>
      <button class="chip" onclick="go('what are the movie clusters')">הצג קבוצות סרטים</button>
    </div>

    <div id="chat" class="chat">
      <div class="msg bot">
        <div class="bubble">ברוכים הבאים לקולנוע החכם 🎞️ כתבו לי מה בא לכם לראות — אפשר לבקש לפי שנה, ז׳אנר או פלטפורמה כמו Netflix.</div>
      </div>
    </div>

    <div class="input-row">
      <input id="inp" placeholder="לדוגמה: סרט אימה משנת 2000 שקיים בנטפליקס..." autocomplete="off">
      <button id="btn">שליחה</button>
    </div>
  </div>
</main>

<script>
const chat = document.getElementById('chat');
const inp = document.getElementById('inp');
const btn = document.getElementById('btn');

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
  results.forEach(r => {{
    h += `<div class="card">
      <div class="card-title">${{esc(r.rank)}}. ${{esc(r.title)}}</div>
      <div class="meta">${{esc(r.year)}} • ⭐ ${{esc(r.rating)}}/10 • התאמה ${{esc(r.score)}}</div>
      <div class="genres">${{esc(r.genres)}}</div>
      ${{r.streaming ? `<div class="stream">זמין ב: ${{esc(r.streaming)}}</div>` : ''}}
      <div class="desc">${{esc(r.overview)}}</div>
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
    headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{message:text}})
  }})
  .then(r => r.json())
  .then(data => {{
    rmTyping();
    let extra = data.intent === 'cluster_info' ? clusters(data.clusters) : cards(data.results);
    add('bot', '<div class="bubble">' + esc(data.reply) + '</div>' + extra);
    if(data.ai_reply){{
      add('bot', '<div class="ai-box">✨ ' + esc(data.ai_reply) + '</div>');
    }}
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
