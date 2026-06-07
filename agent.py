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

app = Flask(__name__)

# ==============================================================
# 1. LOAD & PREPARE DATA - memory optimized for Render Free
# ==============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "movies_master.csv")

# Read only the columns the agent actually uses. This is the biggest memory saver
# when the CSV contains credits/cast/crew or other large columns.
NEEDED_COLUMNS = {
    "title", "overview", "genres", "keywords", "available_on",
    "vote_average", "popularity", "runtime", "vote_count", "release_year",
    "Netflix", "Hulu", "Prime Video", "Disney+"
}

df = pd.read_csv(
    DATA_PATH,
    usecols=lambda c: c in NEEDED_COLUMNS,
    nrows=50000,
    low_memory=False
)

# Make sure optional columns exist even if the CSV does not include them.
for col in NEEDED_COLUMNS:
    if col not in df.columns:
        df[col] = 0 if col in ["vote_average", "popularity", "runtime", "vote_count", "release_year", "Netflix", "Hulu", "Prime Video", "Disney+"] else ""

df = df.reset_index(drop=True)

# Text columns
df["title"] = df["title"].fillna("").astype(str)
df["overview"] = df["overview"].fillna("").astype(str)
df["genres"] = df["genres"].fillna("").astype(str)
df["keywords"] = df["keywords"].fillna("").astype(str)
df["available_on"] = df["available_on"].fillna("לא זמין בסטרימינג").astype(str)

# Numeric columns - float32/int32 save memory compared with defaults
for col in ["vote_average", "popularity", "runtime", "vote_count", "release_year"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")

for col in ["Netflix", "Hulu", "Prime Video", "Disney+"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")

# ==============================================================
# 2. CLUSTERING - sparse matrix, no toarray()
# ==============================================================

from scipy.sparse import hstack, csr_matrix

numeric_features = ["vote_average", "popularity", "runtime", "vote_count"]
scaler = MinMaxScaler()
numeric_scaled = scaler.fit_transform(df[numeric_features]).astype("float32")
numeric_sparse = csr_matrix(numeric_scaled)

def vectorize_column(col, max_features=None):
    vec = CountVectorizer(
        tokenizer=lambda x: [i.strip() for i in str(x).split(",") if i.strip()],
        token_pattern=None,
        binary=True,
        max_features=max_features,
        dtype=np.int8
    )
    return vec.fit_transform(df[col].astype(str))

genres_vec = vectorize_column("genres", max_features=40)
keywords_vec = vectorize_column("keywords", max_features=30)
cluster_data = hstack([numeric_sparse, genres_vec, keywords_vec]).tocsr()

kmeans = MiniBatchKMeans(
    n_clusters=5,
    random_state=42,
    n_init=1,
    batch_size=1024,
    max_iter=40
)
df["cluster"] = kmeans.fit_predict(cluster_data).astype("int8")

# Free memory we no longer need for startup.
del cluster_data

CLUSTER_NAMES = {
    0: "דרמה / פשע / היסטוריה",
    1: "קומדיה / רומנטיקה",
    2: "אקשן / מדע בדיוני / מתח",
    3: "משפחה / אנימציה / פנטזיה",
    4: "אימה / מסתורין / מתח"
}

# ==============================================================
# 3. TF-IDF ON OVERVIEWS - sparse and smaller vocabulary
# ==============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# Do not store another large text column in df; use a temporary Series/list only.
overview_clean = df["overview"].map(clean_text)
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=1800,
    ngram_range=(1, 1),
    dtype=np.float32
)
tfidf_matrix = tfidf.fit_transform(overview_clean)
del overview_clean

# ==============================================================
# 4. SIMILARITY DATA - sparse only, no full similarity matrix
# ==============================================================

sim_data_sparse = hstack([numeric_sparse, genres_vec, keywords_vec]).tocsr()

# ==============================================================
# 5. ANOMALY DETECTION - lighter IsolationForest
# ==============================================================

iso_features = ["popularity", "vote_average", "vote_count", "runtime"]
iso_data = df[iso_features].to_numpy(dtype=np.float32)
iso_scaler = MinMaxScaler()
iso_scaled = iso_scaler.fit_transform(iso_data).astype("float32")

iso = IsolationForest(
    n_estimators=15,
    contamination=0.05,
    random_state=42,
    max_samples=2048
)
df["anomaly"] = iso.fit_predict(iso_scaled).astype("int8")
df["anomaly_score"] = iso.decision_function(iso_scaled).astype("float32")

del iso_data, iso_scaled, iso

# ==============================================================
# 6. GENRE KEYWORD MAP
# ==============================================================

GENRE_KEYWORD_MAP = {
    "action": "Action", "אקשן": "Action", "fight": "Action", "battle": "Action",
    "adventure": "Adventure", "הרפתקה": "Adventure", "quest": "Adventure",
    "animation": "Animation", "אנימציה": "Animation", "cartoon": "Animation",
    "comedy": "Comedy", "קומדיה": "Comedy", "funny": "Comedy", "מצחיק": "Comedy",
    "crime": "Crime", "פשע": "Crime", "heist": "Crime", "detective": "Crime",
    "documentary": "Documentary", "דוקומנטרי": "Documentary",
    "drama": "Drama", "דרמה": "Drama", "emotional": "Drama",
    "family": "Family", "משפחה": "Family", "kids": "Family", "ילדים": "Family",
    "fantasy": "Fantasy", "פנטזיה": "Fantasy", "magic": "Fantasy", "קסם": "Fantasy",
    "history": "History", "היסטוריה": "History", "war": "War", "מלחמה": "War",
    "horror": "Horror", "אימה": "Horror", "scary": "Horror", "מפחיד": "Horror",
    "ghost": "Horror", "רוחות": "Horror", "zombie": "Horror",
    "music": "Music", "מוזיקה": "Music", "musical": "Music",
    "mystery": "Mystery", "מסתורין": "Mystery", "suspense": "Mystery",
    "romance": "Romance", "רומנטי": "Romance", "romantic": "Romance",
    "love": "Romance", "אהבה": "Romance",
    "science fiction": "Science Fiction", "sci-fi": "Science Fiction",
    "מדע בדיוני": "Science Fiction", "space": "Science Fiction",
    "חלל": "Science Fiction", "robot": "Science Fiction", "alien": "Science Fiction",
    "thriller": "Thriller", "מתח": "Thriller", "spy": "Thriller",
    "western": "Western", "קאובוי": "Western",
}

GENRE_TO_CLUSTER = {
    "Action": 2, "Adventure": 2, "Science Fiction": 2, "Thriller": 2,
    "Comedy": 1, "Romance": 1, "Music": 1,
    "Drama": 0, "Crime": 0, "History": 0, "War": 0, "Documentary": 0,
    "Family": 3, "Animation": 3, "Fantasy": 3,
    "Horror": 4, "Mystery": 4,
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

# ==============================================================
# 7. STREAMING BADGE HELPER
# ==============================================================

def get_streaming(row):
    platforms = []
    if row.get("Netflix", 0) == 1:    platforms.append("Netflix")
    if row.get("Hulu", 0) == 1:       platforms.append("Hulu")
    if row.get("Prime Video", 0) == 1: platforms.append("Prime")
    if row.get("Disney+", 0) == 1:    platforms.append("Disney+")
    return ", ".join(platforms) if platforms else ""

# ==============================================================
# 8. INTENT DETECTION
# ==============================================================

def extract_year_filter(text):
    """Extract year from text like 'from 2000' or 'year 2000' or just '2000'"""
    m = re.search(r'\b(19[0-9]{2}|20[0-2][0-9])\b', text)
    return int(m.group(1)) if m else None

def extract_streaming_filter(text):
    """Extract streaming platform filter from text"""
    t = text.lower()
    if any(k in t for k in ["netflix","נטפליקס"]):         return "Netflix"
    if any(k in t for k in ["hulu","הולו"]):               return "Hulu"
    if any(k in t for k in ["prime","פריים","amazon"]):    return "Prime Video"
    if any(k in t for k in ["disney","דיסני"]):            return "Disney+"
    return None

def detect_intent(text):
    t = text.lower()
    for pat in [r"similar to (.+)", r"like (.+)", r"movies like (.+)",
                r"דומה ל(.+)", r"כמו (.+)", r"סרטים כמו (.+)"]:
        m = re.search(pat, t)
        if m:
            return "similar", m.group(1).strip().rstrip("?.,!")
    if any(k in t for k in ["unusual","anomaly","flop","hidden gem","underrated",
                              "פלופ","יהלום נסתר","חריג","מוזר"]):
        return "anomaly", text
    if any(k in t for k in ["cluster","קלסטר","קבוצה","סוגי סרטים"]):
        return "cluster_info", text
    return "search", text

# ==============================================================
# 9. HANDLERS
# ==============================================================

def handle_search(user_text, top_n=3):
    matched_genres   = extract_genres(user_text)
    matched_clusters = genres_to_clusters(matched_genres)
    year_filter      = extract_year_filter(user_text)
    streaming_filter = extract_streaming_filter(user_text)

    cleaned  = clean_text(user_text)
    user_vec = tfidf.transform([cleaned])
    tfidf_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()

    def genre_score(gs):
        if not matched_genres: return 0.0
        mg = [g.strip() for g in gs.split(",")]
        return sum(1 for g in matched_genres if g in mg) / len(matched_genres)

    genre_scores   = df["genres"].apply(genre_score).values
    cluster_scores = df["cluster"].apply(
        lambda c: 1.0 if c in matched_clusters else 0.0).values
    combined = 0.5 * tfidf_scores + 0.3 * genre_scores + 0.2 * cluster_scores

    # Apply year filter
    working_df = df.copy()
    working_combined = combined.copy()
    if year_filter:
        mask = df["release_year"].astype(str).str.startswith(str(year_filter))
        working_combined[~mask.values] = -1

    # Apply streaming filter
    if streaming_filter and streaming_filter in df.columns:
        mask2 = df[streaming_filter] == 1
        working_combined[~mask2.values] = -1

    top_idx = working_combined.argsort()[-top_n:][::-1]
    # Remove filtered-out results
    top_idx = [i for i in top_idx if working_combined[i] > -1][:top_n]

    results = []
    for i, idx in enumerate(top_idx):
        row = df.iloc[idx]
        results.append({
            "rank": i + 1, "title": row["title"],
            "year": int(row["release_year"]) if row["release_year"] else "N/A",
            "genres": row["genres"], "rating": round(float(row["vote_average"]), 1),
            "overview": row["overview"][:200] + "..." if len(row["overview"]) > 200 else row["overview"],
            "score": round(float(combined[idx]), 3),
            "streaming": get_streaming(row)
        })

    reply = "מצאתי " + str(top_n) + " סרטים שמתאימים לבקשה שלך"
    if matched_genres:
        reply += " (ז'אנרים: " + ", ".join(matched_genres) + ")"
    reply += ":"
    return {"intent": "search", "reply": reply, "results": results, "genres": matched_genres}


def handle_similar(movie_title, top_n=3):
    matches = df[df["title"].str.lower().str.contains(movie_title.lower(), na=False)]
    if matches.empty:
        return {"intent": "similar", "reply": "מצטער, לא מצאתי את הסרט '" + movie_title + "' במסד הנתונים.", "results": []}
    idx   = matches.index[0]
    found = df.loc[idx, "title"]
    scores = cosine_similarity(sim_data_sparse[idx], sim_data_sparse).flatten()
    scores[idx] = -1
    top_idx = scores.argsort()[-top_n:][::-1]
    results = []
    for i, tidx in enumerate(top_idx):
        row = df.iloc[tidx]
        results.append({
            "rank": i + 1, "title": row["title"],
            "year": int(row["release_year"]) if row["release_year"] else "N/A",
            "genres": row["genres"], "rating": round(float(row["vote_average"]), 1),
            "overview": row["overview"][:200] + "..." if len(row["overview"]) > 200 else row["overview"],
            "score": round(float(scores[tidx]), 3),
            "streaming": get_streaming(row)
        })
    return {"intent": "similar", "reply": "הנה " + str(top_n) + " סרטים דומים ל-" + found + ":", "results": results}


def handle_anomaly(user_text, top_n=3):
    t = user_text.lower()
    anomalies = df[df["anomaly"] == -1].copy()
    if any(k in t for k in ["flop", "פלופ", "budget", "תקציב"]):
        subset = anomalies[(anomalies["vote_average"] < 5.5) & (anomalies["popularity"] > anomalies["popularity"].median())]
        label  = "פלופים — פופולריים אבל עם ציון נמוך"
    elif any(k in t for k in ["hidden gem", "יהלום נסתר", "underrated", "מוערך בחסר"]):
        subset = anomalies[(anomalies["vote_average"] >= 7.5) &
                           (anomalies["vote_count"] < anomalies["vote_count"].quantile(0.4))]
        label  = "יהלומים נסתרים — ציון גבוה, מעט מכירים"
    elif any(k in t for k in ["long", "short", "ארוך", "קצר", "runtime"]):
        subset = anomalies[(anomalies["runtime"] > 180) | (anomalies["runtime"] < 60)]
        label  = "סרטים עם משך זמן חריג"
    else:
        subset = anomalies.sort_values("anomaly_score").head(top_n)
        label  = "חריגות סטטיסטיות במסד הנתונים"
    subset = subset.head(top_n)
    results = []
    for i, (_, row) in enumerate(subset.iterrows()):
        results.append({
            "rank": i + 1, "title": row["title"],
            "year": int(row["release_year"]) if row["release_year"] else "N/A",
            "genres": row["genres"], "rating": round(float(row["vote_average"]), 1),
            "overview": row["overview"][:200] + "..." if len(row["overview"]) > 200 else row["overview"],
            "score": round(float(row["anomaly_score"]), 3),
            "streaming": get_streaming(row)
        })
    return {"intent": "anomaly", "reply": "הנה " + label + ":", "results": results}


def handle_cluster_info(user_text):
    summary = []
    for c, name in CLUSTER_NAMES.items():
        subset = df[df["cluster"] == c]
        all_g  = ", ".join(subset["genres"].dropna()).split(", ")
        top_g  = [g for g, _ in Counter(all_g).most_common(3) if g]
        summary.append({
            "cluster": c, "name": name, "count": len(subset),
            "top_genres": ", ".join(top_g),
            "avg_rating": round(float(subset["vote_average"].mean()), 2)
        })
    return {"intent": "cluster_info", "reply": "הנה פירוט 5 קלסטרי הסרטים:", "clusters": summary}

# ==============================================================
# 10. OPENAI API
# ==============================================================

def call_openai(user_text, results, intent):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or not results:
        return None
    try:
        import urllib.request
        movies = ""
        for r in results[:5]:
            streaming = (" | זמין ב: " + r["streaming"]) if r.get("streaming") else ""
            movies += "- " + r["title"] + " (" + str(r["year"]) + "): " + r["genres"] + ", " + str(r["rating"]) + "/10" + streaming + "\n"
        prompt = "ענה בעברית בלבד. המשתמש שאל: " + user_text + ". סרטים שנמצאו:\n" + movies + "כתוב 2-3 משפטים ידידותיים שממליצים על הסרטים, ציין 1-2 סרטים בשמם. אם יש מידע על פלטפורמת סטרימינג ציין אותה. טקסט פשוט בלבד ללא markdown."
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
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
# 11. FLASK ROUTES
# ==============================================================

@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")

@app.route("/test-openai")
def test_openai():
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return jsonify({"status": "error", "message": "No OPENAI_API_KEY found"})
    try:
        import urllib.request
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "אמור שלום במשפט אחד בעברית."}],
            "max_tokens": 50
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read())
            return jsonify({"status": "success", "reply": data["choices"][0]["message"]["content"]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_text = data.get("message", "").strip()
    if not user_text:
        return jsonify({"reply": "אנא הקלד משהו!", "results": []})
    intent, payload = detect_intent(user_text)
    if intent == "similar":        result = handle_similar(payload)
    elif intent == "anomaly":      result = handle_anomaly(user_text)
    elif intent == "cluster_info": result = handle_cluster_info(user_text)
    else:                          result = handle_search(user_text)
    if intent != "cluster_info":
        ai_reply = call_openai(user_text, result.get("results", []), intent)
        if ai_reply:
            result["claude_reply"] = ai_reply
    return jsonify(result)

# ==============================================================
# 12. HTML
# ==============================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CineAgent</title>
<link href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700;900&family=Bebas+Neue&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#080808;
  --surface:#0f0f0f;
  --card:#141414;
  --border:#2a0a0a;
  --red:#c0392b;
  --red2:#e74c3c;
  --red3:#ff6b6b;
  --gold:#f5c518;
  --text:#f0e6e6;
  --muted:#806060;
  --dim:#1a0a0a;
}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%}
body{background:var(--bg);color:var(--text);font-family:'Heebo',sans-serif;font-weight:300;direction:rtl;overflow:hidden;display:flex;flex-direction:column;}

/* ---- CINEMA BACKGROUND ---- */
.bg{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}

/* Film grain */
.grain{position:absolute;inset:0;opacity:.04;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")}

/* Spotlight beams */
.beam{position:absolute;top:-10%;width:2px;height:120%;background:linear-gradient(to bottom,transparent,rgba(192,57,43,0.15),transparent);animation:sweepBeam 8s ease-in-out infinite;transform-origin:top center}
.beam1{left:20%;animation-delay:0s}
.beam2{left:50%;animation-delay:2.5s}
.beam3{left:80%;animation-delay:5s}
@keyframes sweepBeam{0%,100%{transform:rotate(-8deg);opacity:.4}50%{transform:rotate(8deg);opacity:.8}}

/* Bottom glow */
.floor-glow{position:absolute;bottom:0;left:0;right:0;height:300px;background:radial-gradient(ellipse 80% 100% at 50% 100%,rgba(192,57,43,0.12) 0%,transparent 70%)}

/* Moving marquee dots */
.marquee-top,.marquee-bottom{position:absolute;left:0;right:0;height:3px;display:flex;gap:0;overflow:hidden}
.marquee-top{top:70px}
.marquee-bottom{bottom:70px}
.mdot{width:8px;height:8px;border-radius:50%;background:var(--red);box-shadow:0 0 8px var(--red2);animation:marqueeDot 2s linear infinite;flex-shrink:0}
.mdot:nth-child(even){background:var(--gold);box-shadow:0 0 8px var(--gold);animation-delay:.2s}
@keyframes marqueeDot{0%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.6)}100%{opacity:1;transform:scale(1)}}

/* Curtain sides */
.curtain-l,.curtain-r{position:absolute;top:0;bottom:0;width:60px;z-index:1}
.curtain-l{left:0;background:linear-gradient(to right,rgba(60,0,0,0.6),transparent)}
.curtain-r{right:0;background:linear-gradient(to left,rgba(60,0,0,0.6),transparent)}

/* ---- HEADER ---- */
header{position:relative;z-index:10;display:flex;align-items:center;justify-content:space-between;padding:14px 32px;border-bottom:1px solid rgba(192,57,43,0.3);background:rgba(8,8,8,0.95);backdrop-filter:blur(10px);flex-shrink:0}
.logo-wrap{display:flex;align-items:center;gap:12px}
.logo-icon{font-size:1.6rem;filter:drop-shadow(0 0 8px var(--red))}
.logo-text{font-family:'Bebas Neue',sans-serif;font-size:1.8rem;letter-spacing:4px;color:var(--text);line-height:1}
.logo-text span{color:var(--red2)}
.logo-sub{font-size:.6rem;color:var(--muted);letter-spacing:3px;text-transform:uppercase;margin-top:2px}
.badge{display:flex;align-items:center;gap:6px;background:rgba(192,57,43,0.1);border:1px solid rgba(192,57,43,0.3);border-radius:20px;padding:5px 14px;font-size:.68rem;color:var(--red3);font-weight:500}
.bdot{width:6px;height:6px;border-radius:50%;background:var(--red2);box-shadow:0 0 6px var(--red2);animation:rpulse 2s infinite}
@keyframes rpulse{0%,100%{opacity:1;box-shadow:0 0 6px var(--red2)}50%{opacity:.4;box-shadow:0 0 12px var(--red2)}}

/* ---- MAIN ---- */
.main{position:relative;z-index:5;flex:1;display:flex;flex-direction:column;overflow:hidden}

/* ---- SUGGESTIONS BAR ---- */
.suggestions-bar{padding:12px 24px;border-bottom:1px solid rgba(192,57,43,0.15);background:rgba(10,0,0,0.6);display:flex;gap:8px;flex-wrap:wrap;justify-content:center;flex-shrink:0}
.sug{background:rgba(192,57,43,0.08);border:1px solid rgba(192,57,43,0.2);border-radius:20px;padding:6px 14px;font-size:.75rem;color:var(--muted);cursor:pointer;transition:all .2s;white-space:nowrap;font-family:'Heebo',sans-serif}
.sug:hover{border-color:var(--red2);color:var(--text);background:rgba(192,57,43,0.15);box-shadow:0 0 10px rgba(192,57,43,0.2)}
.sug-tag{color:var(--red3);font-weight:700;margin-left:4px;font-size:.65rem}

/* ---- MESSAGES ---- */
#messages{flex:1;overflow-y:auto;padding:20px 15%;display:flex;flex-direction:column;gap:16px;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
@media(max-width:900px){#messages{padding:16px 4%}}

.msg{display:flex;flex-direction:column;gap:8px;animation:msgIn .3s ease}
@keyframes msgIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.msg.user{align-items:flex-start}
.msg.bot{align-items:flex-end}

.bubble{max-width:580px;padding:12px 18px;border-radius:4px;font-size:.88rem;line-height:1.7}
.msg.user .bubble{background:linear-gradient(135deg,#6b0000,#c0392b);color:#fff;border-bottom-left-radius:16px;box-shadow:0 4px 20px rgba(192,57,43,0.3)}
.msg.bot .bubble{background:var(--card);border:1px solid rgba(192,57,43,0.2);border-bottom-right-radius:16px;box-shadow:0 2px 12px rgba(0,0,0,0.4)}
.bubble b{color:var(--gold)}

/* AI box */
.ai-box{max-width:580px;padding:14px 18px;border-radius:4px;border-bottom-right-radius:16px;background:#0a0000;border:1px solid rgba(245,197,24,0.25);font-size:.85rem;line-height:1.75;color:#c8b090;position:relative;overflow:hidden}
.ai-box::before{content:'';position:absolute;top:0;right:0;left:0;height:1px;background:linear-gradient(90deg,transparent,rgba(245,197,24,0.5),transparent)}
.ai-tag{font-size:.58rem;letter-spacing:2px;text-transform:uppercase;color:var(--gold);margin-bottom:8px;font-weight:700;display:flex;align-items:center;gap:5px}
.ai-tag::before{content:"★";font-size:.7rem}

/* Cards */
.cards{display:flex;flex-direction:column;gap:8px;width:100%;max-width:580px}
.card{background:var(--card);border:1px solid rgba(192,57,43,0.15);border-right:3px solid var(--red);padding:12px 14px;display:flex;gap:12px;transition:all .2s;border-radius:2px}
.card:hover{border-color:rgba(192,57,43,0.5);border-right-color:var(--red2);transform:translateX(4px);box-shadow:0 4px 20px rgba(192,57,43,0.15)}
.cnum{font-family:'Bebas Neue',sans-serif;font-size:2rem;color:rgba(192,57,43,0.3);line-height:1;min-width:26px;text-align:center;transition:color .2s}
.card:hover .cnum{color:var(--red2)}
.cbody{flex:1;min-width:0}
.ctitle{font-size:.9rem;font-weight:700;margin-bottom:3px;color:var(--text)}
.cmeta{font-size:.67rem;color:var(--muted);display:flex;gap:8px;flex-wrap:wrap;margin-bottom:4px;align-items:center}
.cgenres{font-size:.67rem;color:rgba(192,57,43,0.8);margin-bottom:4px;font-weight:500}
.cstream{font-size:.63rem;margin-bottom:4px}
.sp{display:inline-block;padding:2px 6px;border-radius:3px;margin-left:3px;font-weight:600;font-size:.6rem}
.sp-netflix{background:rgba(229,9,20,0.2);color:#ff4444;border:1px solid rgba(229,9,20,0.3)}
.sp-hulu{background:rgba(28,231,131,0.1);color:#1ce783;border:1px solid rgba(28,231,131,0.25)}
.sp-prime{background:rgba(0,168,225,0.1);color:#00a8e1;border:1px solid rgba(0,168,225,0.25)}
.sp-disney{background:rgba(17,60,219,0.15);color:#5577ff;border:1px solid rgba(17,60,219,0.3)}
.cdesc{font-size:.73rem;color:#604040;line-height:1.5}
.cscore{font-size:.58rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;white-space:nowrap;text-align:center}
.rdot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-left:3px}

/* Clusters */
.clusters{display:flex;flex-wrap:wrap;gap:8px;max-width:580px}
.clu{background:var(--card);border:1px solid rgba(192,57,43,0.2);border-top:2px solid var(--red);padding:12px 14px;flex:1;min-width:140px}
.clu-name{font-size:.82rem;font-weight:700;color:var(--red3);margin-bottom:5px}
.clu-stat{font-size:.68rem;color:var(--muted);line-height:1.8}

/* Typing */
.typing{display:flex;gap:5px;padding:12px 16px;background:var(--card);border:1px solid rgba(192,57,43,0.2);border-radius:4px;border-bottom-right-radius:16px;width:fit-content}
.dot{width:6px;height:6px;background:var(--muted);border-radius:50%;animation:tdot 1.2s infinite}
.dot:nth-child(2){animation-delay:.2s}.dot:nth-child(3){animation-delay:.4s}
@keyframes tdot{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-6px);background:var(--red2);box-shadow:0 0 6px var(--red2)}}

/* ---- INPUT BAR ---- */
.input-wrap{position:relative;z-index:10;padding:14px 15%;background:rgba(8,8,8,0.95);backdrop-filter:blur(10px);border-top:1px solid rgba(192,57,43,0.2);flex-shrink:0}
@media(max-width:900px){.input-wrap{padding:12px 4%}}
.input-inner{display:flex;align-items:center;background:var(--dim);border:1px solid rgba(192,57,43,0.3);border-radius:4px;padding:6px 6px 6px 14px;transition:border-color .2s;box-shadow:inset 0 0 20px rgba(0,0,0,0.5)}
.input-inner:focus-within{border-color:var(--red2);box-shadow:inset 0 0 20px rgba(0,0,0,0.5),0 0 0 2px rgba(192,57,43,0.15)}
#inp{flex:1;background:transparent;border:none;outline:none;color:var(--text);font-family:'Heebo',sans-serif;font-size:.9rem;direction:rtl;padding:8px 4px}
#inp::placeholder{color:var(--muted)}
#btn{background:linear-gradient(135deg,#8b0000,#c0392b);color:#fff;border:none;border-radius:3px;width:44px;height:44px;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:1.1rem;transition:all .2s;flex-shrink:0;box-shadow:0 2px 12px rgba(192,57,43,0.4)}
#btn:hover{background:linear-gradient(135deg,#a00000,#e74c3c);box-shadow:0 4px 20px rgba(192,57,43,0.6);transform:scale(1.05)}
#btn:active{transform:scale(.95)}
.input-hint{text-align:center;margin-top:6px;font-size:.6rem;color:var(--muted)}
</style>
</head>
<body>

<!-- CINEMA BACKGROUND -->
<div class="bg">
  <div class="grain"></div>
  <div class="beam beam1"></div>
  <div class="beam beam2"></div>
  <div class="beam beam3"></div>
  <div class="floor-glow"></div>
  <div class="curtain-l"></div>
  <div class="curtain-r"></div>
  <div class="marquee-top" id="mt"></div>
  <div class="marquee-bottom" id="mb"></div>
</div>

<header>
  <div class="logo-wrap">
    <div class="logo-icon">&#127902;</div>
    <div>
      <div class="logo-text">CINE<span>AGENT</span></div>
      <div class="logo-sub">AI Movie Intelligence</div>
    </div>
  </div>
  <div class="badge"><div class="bdot"></div>פעיל</div>
</header>

<!-- SUGGESTIONS BAR -->
<div class="suggestions-bar">
  <div class="sug" onclick="go('קומדיה רומנטית מצחיקה')"><span class="sug-tag">חיפוש</span>קומדיה רומנטית</div>
  <div class="sug" onclick="go('סרט אימה עם רוחות')"><span class="sug-tag">חיפוש</span>סרט אימה</div>
  <div class="sug" onclick="go('movies similar to Inception')"><span class="sug-tag">דומה</span>דומה ל-Inception</div>
  <div class="sug" onclick="go('סרט אקשן בנטפליקס')"><span class="sug-tag">סטרימינג</span>אקשן בנטפליקס</div>
  <div class="sug" onclick="go('סרט אנימציה לילדים בדיסני')"><span class="sug-tag">סטרימינג</span>אנימציה בדיסני+</div>
  <div class="sug" onclick="go('find me hidden gems with high rating')"><span class="sug-tag">חריגה</span>יהלומים נסתרים</div>
  <div class="sug" onclick="go('סרט דרמה משנת 2010')"><span class="sug-tag">שנה</span>דרמה משנת 2010</div>
  <div class="sug" onclick="go('what are the movie clusters')"><span class="sug-tag">קלסטרים</span>קלסטרי סרטים</div>
</div>

<div class="main">
  <div id="messages">
    <div class="msg bot">
      <div class="bubble">
        &#127916; ברוכים הבאים ל<b>CineAgent</b>!<br><br>
        אני משלב <b>למידת מכונה</b>, <b>NLP</b> ו<b>ChatGPT AI</b> למציאת סרטים.<br>
        מסד נתונים של <b>50,000 סרטים</b> עם מידע על Netflix, Hulu, Prime ו-Disney+.<br><br>
        ניתן לסנן לפי <b>שנה</b> (למשל: "סרט משנת 2005") או לפי <b>פלטפורמה</b> (למשל: "סרט בנטפליקס").
      </div>
    </div>
  </div>
  <div class="input-wrap">
    <div class="input-inner">
      <input id="inp" type="text" placeholder="חפש סרט, תאר מצב רוח, או בקש המלצה..." autocomplete="off" />
      <button id="btn">&#x25B6;</button>
    </div>
    <div class="input-hint">Enter &#x21B5; לשליחה</div>
  </div>
</div>

<script>
// Generate marquee dots
var mt=document.getElementById('mt'), mb=document.getElementById('mb');
for(var i=0;i<60;i++){
  var d1=document.createElement('div'); d1.className='mdot';
  d1.style.animationDelay=(i*0.15)+'s'; mt.appendChild(d1);
  var d2=document.createElement('div'); d2.className='mdot';
  d2.style.animationDelay=(i*0.15+0.5)+'s'; mb.appendChild(d2);
}

var M=document.getElementById('messages'),I=document.getElementById('inp'),B=document.getElementById('btn');

function rc(r){return r>=7.5?'#2ecc71':r>=6?'#f39c12':'#e74c3c';}

function streamBadges(s){
  if(!s)return '';
  var h='',parts=s.split(', ');
  for(var i=0;i<parts.length;i++){
    var p=parts[i].trim();
    if(p==='Netflix')h+='<span class="sp sp-netflix">Netflix</span>';
    else if(p==='Hulu')h+='<span class="sp sp-hulu">Hulu</span>';
    else if(p==='Prime')h+='<span class="sp sp-prime">Prime</span>';
    else if(p==='Disney+')h+='<span class="sp sp-disney">Disney+</span>';
  }
  return h?'<div class="cstream">'+h+'</div>':'';
}

function addMsg(role,html){
  var d=document.createElement('div');d.className='msg '+role;d.innerHTML=html;
  M.appendChild(d);M.scrollTop=M.scrollHeight;
}

function addTyping(){
  var d=document.createElement('div');d.className='msg bot';d.id='typ';
  d.innerHTML='<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
  M.appendChild(d);M.scrollTop=M.scrollHeight;
}
function rmTyping(){var t=document.getElementById('typ');if(t)t.remove();}

function buildCards(results){
  if(!results||!results.length)return '';
  var h='<div class="cards">';
  for(var i=0;i<results.length;i++){
    var r=results[i];
    h+='<div class="card">';
    h+='<div class="cnum">'+r.rank+'</div>';
    h+='<div class="cbody">';
    h+='<div class="ctitle">'+r.title+'</div>';
    h+='<div class="cmeta"><span>'+r.year+'</span><span><span class="rdot" style="background:'+rc(r.rating)+'"></span>'+r.rating+'/10</span></div>';
    h+='<div class="cgenres">'+r.genres+'</div>';
    h+=streamBadges(r.streaming);
    h+='<div class="cdesc">'+r.overview+'</div>';
    h+='</div><div class="cscore">ציון<br>'+r.score+'</div></div>';
  }
  return h+'</div>';
}

function buildClusters(clusters){
  if(!clusters)return '';
  var h='<div class="clusters">';
  for(var i=0;i<clusters.length;i++){
    var c=clusters[i];
    h+='<div class="clu"><div class="clu-name">'+c.name+'</div>';
    h+='<div class="clu-stat">'+c.count+' סרטים | ממוצע: '+c.avg_rating+'<br>'+c.top_genres+'</div></div>';
  }
  return h+'</div>';
}

function send(){
  var text=I.value.trim();if(!text)return;
  addMsg('user','<div class="bubble">'+text+'</div>');
  I.value='';addTyping();
  fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})})
  .then(function(r){return r.json();})
  .then(function(data){
    rmTyping();
    var extra=data.intent==='cluster_info'?buildClusters(data.clusters):buildCards(data.results);
    addMsg('bot','<div class="bubble">'+data.reply+'</div>'+extra);
    if(data.claude_reply){
      setTimeout(function(){
        addMsg('bot','<div class="ai-box"><div class="ai-tag">ChatGPT AI</div>'+data.claude_reply+'</div>');
        M.scrollTop=M.scrollHeight;
      },300);
    }
  })
  .catch(function(e){rmTyping();addMsg('bot','<div class="bubble">משהו השתבש. נסה שוב.</div>');});
}

function go(q){I.value=q;send();}
B.onclick=function(){send();};
I.onkeydown=function(e){if(e.key==='Enter')send();};
</script>
</body>
</html>"""

# ==============================================================
# 13. RUN
# ==============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
