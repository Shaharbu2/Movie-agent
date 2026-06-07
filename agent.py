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

def handle_search(user_text, top_n=5):
    matched_genres   = extract_genres(user_text)
    matched_clusters = genres_to_clusters(matched_genres)
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
    top_idx  = combined.argsort()[-top_n:][::-1]

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


def handle_similar(movie_title, top_n=5):
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


def handle_anomaly(user_text, top_n=6):
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
<title>Movie Chatbot</title>
<link href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700;900&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{
  direction:rtl;font-family:'Heebo',Arial,sans-serif;color:#2f2f38;min-height:100vh;
  background:
    linear-gradient(rgba(255,255,255,.38),rgba(255,255,255,.45)),
    radial-gradient(circle at 15% 20%,rgba(255,210,95,.55),transparent 22%),
    radial-gradient(circle at 80% 10%,rgba(238,64,95,.32),transparent 24%),
    linear-gradient(135deg,#141821 0%,#33354b 45%,#0f1118 100%);
  background-attachment:fixed;
}
body:before{
  content:"";position:fixed;inset:80px 0 0 0;z-index:-1;opacity:.34;
  background-image:
    linear-gradient(120deg,rgba(0,0,0,.65),rgba(0,0,0,.1)),
    url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=1800&q=80');
  background-size:cover;background-position:center;
}
.topbar{
  height:80px;background:rgba(255,255,255,.96);display:flex;align-items:center;justify-content:space-between;
  padding:0 42px;box-shadow:0 2px 16px rgba(0,0,0,.12);position:sticky;top:0;z-index:10
}
.brand{font-size:31px;font-weight:900;letter-spacing:.3px;color:#303038;display:flex;gap:10px;align-items:center}
.brand .icon{font-size:31px}.status{font-size:16px;font-weight:700;color:#777;background:#f4f4f8;border-radius:999px;padding:8px 16px}
.hero{text-align:center;padding:34px 18px 30px;background:rgba(255,255,255,.72);backdrop-filter:blur(3px)}
.hero h1{font-size:clamp(38px,5vw,68px);line-height:1.05;font-weight:900;color:#303038;text-shadow:0 2px 0 rgba(255,255,255,.7)}
.hero p{font-size:clamp(20px,2.2vw,30px);margin-top:22px;color:#464653;font-weight:400}
.intro-card{
  width:min(1400px,86vw);margin:46px auto 30px;background:rgba(255,255,255,.88);border-radius:22px;
  padding:58px 64px;box-shadow:0 12px 38px rgba(20,20,30,.14);backdrop-filter:blur(5px)
}
.intro-card h2{font-size:30px;margin-bottom:28px;color:#2f2f38;font-weight:900}
.intro-card p{font-size:24px;line-height:1.9;color:#444450}.gift{margin-left:12px}
.start-btn{margin-top:32px;background:#ef3d7a;color:#fff;border:0;border-radius:38px;padding:20px 46px;font-size:25px;font-weight:900;cursor:pointer;box-shadow:0 8px 24px rgba(239,61,122,.25);transition:.2s}
.start-btn:hover{transform:translateY(-2px);background:#e32969}
.examples{margin-top:44px}.examples h3{font-size:27px;font-weight:900;margin-bottom:20px}
.examples ul{font-size:23px;line-height:2.1;padding-right:25px}.examples li{padding-right:8px}.examples span{margin-left:10px}
.chat-shell{width:min(1350px,86vw);margin:34px auto 80px;background:rgba(255,255,255,.96);border-radius:20px;box-shadow:0 12px 38px rgba(20,20,30,.15);overflow:hidden;min-height:650px}
.chat-head{height:76px;display:flex;align-items:center;justify-content:space-between;padding:0 30px;border-bottom:1px solid #eee;font-size:22px;font-weight:900;color:#0f0f12;background:#fff}
.dots{letter-spacing:3px;color:#777}.messages{height:470px;overflow-y:auto;padding:34px 42px;display:flex;flex-direction:column;gap:24px;background:rgba(255,255,255,.72)}
.msg{display:flex;flex-direction:column;max-width:78%;animation:in .22s ease}.msg.user{align-self:flex-start}.msg.bot{align-self:flex-end}
.bubble{font-size:22px;line-height:1.7;border-radius:27px;padding:20px 28px;white-space:pre-wrap}
.user .bubble{background:#3f7df0;color:white;border-bottom-left-radius:8px;font-weight:700}
.bot .bubble{background:#f0f0f3;color:#2e2e38;border-bottom-right-radius:8px}.time{font-size:15px;color:#8a8a92;margin-top:8px}
.cards{display:flex;flex-direction:column;gap:12px;margin-top:12px}.movie-card{background:#fff;border:1px solid #eee;border-radius:18px;padding:16px 20px;box-shadow:0 4px 18px rgba(0,0,0,.05)}
.movie-title{font-size:21px;font-weight:900;color:#222;margin-bottom:6px}.movie-meta{font-size:16px;color:#6c6c75;margin-bottom:5px}.movie-genres{font-size:16px;color:#e33d75;font-weight:800;margin-bottom:6px}.movie-desc{font-size:16px;line-height:1.55;color:#555}
.clusters{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-top:12px}.cluster{background:#fff;border:1px solid #eee;border-radius:16px;padding:16px}.cluster b{color:#e33d75}
.input-zone{display:flex;gap:12px;align-items:center;padding:22px 30px;border-top:1px solid #eee;background:#fff}
#inp{flex:1;border:1px solid #e1e1e7;border-radius:28px;padding:18px 24px;font-size:20px;font-family:inherit;outline:none;background:#fafafa}
#inp:focus{border-color:#ef3d7a;box-shadow:0 0 0 4px rgba(239,61,122,.08)}
#btn{border:0;background:#ef3d7a;color:#fff;border-radius:28px;padding:17px 31px;font-size:20px;font-weight:900;cursor:pointer}#btn:hover{background:#e32969}
.quick{display:flex;flex-wrap:wrap;gap:10px;padding:0 30px 24px;background:#fff}.chip{border:1px solid #eee;background:#f8f8fb;border-radius:999px;padding:10px 16px;font-size:16px;cursor:pointer}.chip:hover{border-color:#ef3d7a;color:#ef3d7a}
@keyframes in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@media(max-width:760px){.topbar{padding:0 18px}.status{display:none}.intro-card,.chat-shell{width:94vw}.intro-card{padding:32px 24px}.messages{height:430px;padding:24px 18px}.msg{max-width:94%}.bubble{font-size:18px}.hero h1{font-size:42px}.hero p,.intro-card p,.examples ul{font-size:19px}}
</style>
</head>
<body>
<header class="topbar">
  <div class="brand"><span class="icon">🎬</span> צ׳אטבוט סרטים</div>
  <div class="status">מבוסס AI + 50,000 סרטים</div>
</header>
<section class="hero">
  <h1>מחפשים את הסרט המושלם? 🍿</h1>
  <p>הבוט שלנו יענה על כל שאלה על סרטים, המלצות, ז׳אנרים וסרטים דומים</p>
</section>
<section class="intro-card">
  <h2>איך אפשר לעזור לכם היום?</h2>
  <p><span class="gift">🎁</span>הבוט יודע להתאים סרט לכל מצב רוח, למצוא סרטים דומים, לזהות חריגות מעניינות בדאטה, ולהציג קבוצות סרטים שנמצאו בעזרת למידת מכונה.</p>
  <button class="start-btn" onclick="document.getElementById('chat').scrollIntoView({behavior:'smooth'})">התחילו לדבר עם הבוט</button>
  <div class="examples">
    <h3>דוגמאות לשאלות שתוכלו לשאול:</h3>
    <ul>
      <li><span>🎭</span>אני רוצה קומדיה רומנטית מצחיקה</li>
      <li><span>🚀</span>תמצא לי סרט מדע בדיוני כמו Inception</li>
      <li><span>💎</span>find me hidden gems with high rating</li>
      <li><span>📊</span>what are the movie clusters?</li>
    </ul>
  </div>
</section>
<section class="chat-shell" id="chat">
  <div class="chat-head"><span>movie_chatbot</span><span class="dots">•••</span></div>
  <div class="messages" id="messages">
    <div class="msg bot"><div class="bubble">ברוכים הבאים לצ׳אטבוט הסרטים! 🌟\nאני כאן כדי לעזור לכם לבחור סרט לפי מצב רוח, ז׳אנר, סרט דומה או חריגות מעניינות. כתבו לי מה אתם מחפשים — ואני אמצא לכם המלצה מדויקת.</div></div>
  </div>
  <div class="input-zone">
    <input id="inp" placeholder="כתבו כאן שאלה על סרטים..." autocomplete="off">
    <button id="btn">שליחה</button>
  </div>
  <div class="quick">
    <button class="chip" onclick="go('אני רוצה קומדיה רומנטית מצחיקה')">קומדיה רומנטית</button>
    <button class="chip" onclick="go('movies similar to Inception')">דומה ל־Inception</button>
    <button class="chip" onclick="go('show me big budget flops')">פלופים</button>
    <button class="chip" onclick="go('what are the movie clusters')">קלסטרים</button>
  </div>
</section>
<script>
const M=document.getElementById('messages'), I=document.getElementById('inp'), B=document.getElementById('btn');
function esc(s){return String(s||'').replace(/[&<>"]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m]));}
function addMsg(role,html){const d=document.createElement('div');d.className='msg '+role;d.innerHTML=html;M.appendChild(d);M.scrollTop=M.scrollHeight;}
function cards(results){if(!results||!results.length)return '';let h='<div class="cards">';results.forEach(r=>{h+=`<div class="movie-card"><div class="movie-title">${esc(r.rank)}. ${esc(r.title)}</div><div class="movie-meta">${esc(r.year)} | דירוג ${esc(r.rating)}/10 | ציון התאמה ${esc(r.score)}</div><div class="movie-genres">${esc(r.genres)}</div>${r.streaming?`<div class="movie-meta">זמין ב: ${esc(r.streaming)}</div>`:''}<div class="movie-desc">${esc(r.overview)}</div></div>`});return h+'</div>';}
function clusters(cs){if(!cs)return '';let h='<div class="clusters">';cs.forEach(c=>{h+=`<div class="cluster"><b>${esc(c.name)}</b><br>${esc(c.count)} סרטים<br>דירוג ממוצע: ${esc(c.avg_rating)}<br>${esc(c.top_genres)}</div>`});return h+'</div>';}
function send(){let text=I.value.trim();if(!text)return;addMsg('user',`<div class="bubble">${esc(text)}</div>`);I.value='';addMsg('bot','<div class="bubble">חושבת על המלצה טובה... 🍿</div>');let loading=M.lastChild;fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})}).then(r=>r.json()).then(data=>{loading.remove();let extra=data.intent==='cluster_info'?clusters(data.clusters):cards(data.results);addMsg('bot',`<div class="bubble">${esc(data.reply)}</div>${extra}`);if(data.claude_reply){addMsg('bot',`<div class="bubble">${esc(data.claude_reply)}</div>`);}}).catch(()=>{loading.remove();addMsg('bot','<div class="bubble">משהו השתבש, נסו שוב בעוד רגע.</div>');});}
function go(q){I.value=q;document.getElementById('chat').scrollIntoView({behavior:'smooth'});send();}
B.onclick=send;I.addEventListener('keydown',e=>{if(e.key==='Enter')send();});
</script>
</body>
</html>"""

# ==============================================================
# 13. RUN
# ==============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
