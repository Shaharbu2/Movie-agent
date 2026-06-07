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
<title>CineAgent</title>
<link href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700;900&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#060914;--surface:#0b0d1a;--card:#111320;--border:#1e2140;
  --gold:#e8b84b;--gold2:#f5d07a;--text:#dde2ff;--muted:#4a5080;
  --accent:#3d6fff;--green:#3de0a0;--ai:#0e1128;
}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%}
body{background:var(--bg);color:var(--text);font-family:'Heebo',sans-serif;font-weight:300;direction:rtl;overflow:hidden;display:flex;flex-direction:column;}
.bg-wrap{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}
.bg-grad{position:absolute;inset:0;background:radial-gradient(ellipse 70% 50% at 50% -10%,rgba(61,111,255,0.18) 0%,transparent 60%),radial-gradient(ellipse 40% 40% at 85% 80%,rgba(232,184,75,0.07) 0%,transparent 55%)}
.grid{position:absolute;inset:0;background-image:linear-gradient(rgba(61,111,255,0.04) 1px,transparent 1px),linear-gradient(90deg,rgba(61,111,255,0.04) 1px,transparent 1px);background-size:60px 60px}
.orb{position:absolute;border-radius:50%;filter:blur(80px);animation:float 8s ease-in-out infinite}
.orb1{width:400px;height:400px;background:rgba(61,111,255,0.08);top:-100px;left:50%;transform:translateX(-50%)}
.orb2{width:300px;height:300px;background:rgba(232,184,75,0.05);bottom:100px;right:10%;animation-delay:3s}
@keyframes float{0%,100%{transform:translateY(0) translateX(-50%)}50%{transform:translateY(-30px) translateX(-50%)}}
header{position:relative;z-index:10;display:flex;align-items:center;justify-content:space-between;padding:16px 32px;border-bottom:1px solid rgba(61,111,255,0.15);background:rgba(6,9,20,0.8);backdrop-filter:blur(20px);flex-shrink:0}
.logo-wrap{display:flex;align-items:center;gap:14px}
.logo-icon{width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,#1a2a6c,#3d6fff);border:1px solid rgba(61,111,255,0.4);display:flex;align-items:center;justify-content:center;font-size:1.2rem;box-shadow:0 0 20px rgba(61,111,255,0.3)}
.logo-text{font-size:1.3rem;font-weight:900;color:var(--text);letter-spacing:1px}
.logo-text span{color:var(--gold)}
.logo-sub{font-size:.65rem;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-top:1px}
.badge{display:flex;align-items:center;gap:6px;background:rgba(61,224,160,0.08);border:1px solid rgba(61,224,160,0.2);border-radius:20px;padding:5px 14px;font-size:.68rem;color:var(--green);font-weight:500}
.bdot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.main{position:relative;z-index:5;flex:1;display:flex;flex-direction:column;overflow:hidden}
#welcome{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:40px 20px;text-align:center;gap:28px}
.wicon{width:90px;height:90px;border-radius:24px;background:linear-gradient(135deg,#0d1a4a,#1a3a8f);border:1px solid rgba(61,111,255,0.35);display:flex;align-items:center;justify-content:center;font-size:2.5rem;box-shadow:0 0 60px rgba(61,111,255,0.25)}
.wtitle{font-size:clamp(2rem,5vw,3.2rem);font-weight:900;background:linear-gradient(135deg,#dde2ff 0%,#e8b84b 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1}
.wsub{font-size:.95rem;color:var(--muted);line-height:1.8;max-width:520px}
.wsub b{color:var(--text);font-weight:500}
.chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;max-width:640px}
.chip{background:rgba(255,255,255,0.04);border:1px solid var(--border);border-radius:24px;padding:8px 16px;font-size:.78rem;color:var(--muted);cursor:pointer;transition:all .2s;white-space:nowrap}
.chip:hover{border-color:var(--accent);color:var(--text);background:rgba(61,111,255,0.08)}
.ctag{color:var(--gold);font-weight:700;margin-left:4px;font-size:.65rem}
#chat-screen{flex:1;display:none;flex-direction:column;overflow:hidden}
#chat-screen.active{display:flex}
#messages{flex:1;overflow-y:auto;padding:24px 20%;display:flex;flex-direction:column;gap:20px;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
@media(max-width:900px){#messages{padding:20px 5%}}
.msg{display:flex;flex-direction:column;gap:8px;animation:msgIn .3s ease}
@keyframes msgIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.msg.user{align-items:flex-start}.msg.bot{align-items:flex-end}
.bubble{max-width:600px;padding:13px 18px;border-radius:18px;font-size:.9rem;line-height:1.7}
.msg.user .bubble{background:linear-gradient(135deg,#1a3080,#3d6fff);color:#fff;border-bottom-left-radius:4px;box-shadow:0 4px 20px rgba(61,111,255,0.3)}
.msg.bot .bubble{background:var(--card);border:1px solid var(--border);border-bottom-right-radius:4px}
.bubble b{color:var(--gold)}
.ai-box{max-width:600px;padding:16px 18px;border-radius:18px;border-bottom-right-radius:4px;background:var(--ai);border:1px solid rgba(232,184,75,0.2);font-size:.87rem;line-height:1.75;color:#b8c0e8;position:relative;overflow:hidden}
.ai-box::before{content:'';position:absolute;top:0;right:0;left:0;height:1px;background:linear-gradient(90deg,transparent,rgba(232,184,75,0.4),transparent)}
.ai-tag{font-size:.6rem;letter-spacing:2px;text-transform:uppercase;color:var(--gold);margin-bottom:10px;font-weight:700;display:flex;align-items:center;gap:6px}
.ai-tag::before{content:"✦";font-size:.75rem}
.cards{display:flex;flex-direction:column;gap:10px;width:100%;max-width:600px}
.card{background:rgba(17,19,32,0.8);border:1px solid var(--border);border-radius:14px;padding:14px 16px;display:flex;gap:14px;transition:all .2s}
.card:hover{border-color:rgba(61,111,255,0.4);transform:translateX(4px);box-shadow:0 4px 24px rgba(61,111,255,0.1)}
.cnum{font-size:1.8rem;font-weight:900;color:var(--border);line-height:1;min-width:28px;text-align:center;transition:color .2s}
.card:hover .cnum{color:var(--gold)}
.cbody{flex:1;min-width:0}
.ctitle{font-size:.92rem;font-weight:700;margin-bottom:4px;color:var(--text)}
.cmeta{font-size:.68rem;color:var(--muted);display:flex;gap:8px;flex-wrap:wrap;margin-bottom:5px;align-items:center}
.cgenres{font-size:.68rem;color:var(--gold);margin-bottom:4px;font-weight:500}
.cstream{font-size:.65rem;margin-bottom:5px}
.sp{display:inline-block;padding:2px 7px;border-radius:10px;margin-left:3px;font-weight:600}
.sp-netflix{background:rgba(229,9,20,0.2);color:#e55;border:1px solid rgba(229,9,20,0.3)}
.sp-hulu{background:rgba(28,231,131,0.15);color:#1ce783;border:1px solid rgba(28,231,131,0.3)}
.sp-prime{background:rgba(0,168,225,0.15);color:#00a8e1;border:1px solid rgba(0,168,225,0.3)}
.sp-disney{background:rgba(17,60,219,0.2);color:#5577ff;border:1px solid rgba(17,60,219,0.3)}
.cdesc{font-size:.75rem;color:#5a6090;line-height:1.55}
.cscore{font-size:.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;white-space:nowrap;text-align:center}
.rdot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-left:3px}
.clusters{display:flex;flex-wrap:wrap;gap:8px;max-width:600px}
.clu{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px 16px;flex:1;min-width:150px}
.clu-name{font-size:.85rem;font-weight:700;color:var(--gold);margin-bottom:6px}
.clu-stat{font-size:.7rem;color:var(--muted);line-height:1.8}
.typing{display:flex;gap:5px;padding:14px 18px;background:var(--card);border:1px solid var(--border);border-radius:18px;border-bottom-right-radius:4px;width:fit-content}
.dot{width:6px;height:6px;background:var(--muted);border-radius:50%;animation:b 1.2s infinite}
.dot:nth-child(2){animation-delay:.2s}.dot:nth-child(3){animation-delay:.4s}
@keyframes b{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-6px);background:var(--gold)}}
.input-wrap{position:relative;z-index:10;padding:20px 20%;background:rgba(6,9,20,0.7);backdrop-filter:blur(20px);border-top:1px solid rgba(61,111,255,0.1);flex-shrink:0}
@media(max-width:900px){.input-wrap{padding:16px 5%}}
.input-inner{display:flex;align-items:center;background:rgba(17,19,32,0.9);border:1px solid rgba(61,111,255,0.25);border-radius:16px;padding:6px 6px 6px 16px;transition:border-color .2s}
.input-inner:focus-within{border-color:rgba(61,111,255,0.6);box-shadow:0 0 0 3px rgba(61,111,255,0.1)}
#inp{flex:1;background:transparent;border:none;outline:none;color:var(--text);font-family:'Heebo',sans-serif;font-size:.92rem;direction:rtl;padding:8px 4px}
#inp::placeholder{color:var(--muted)}
#btn{background:linear-gradient(135deg,#2a50d0,#3d6fff);color:#fff;border:none;border-radius:11px;width:44px;height:44px;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:1.1rem;transition:all .2s;flex-shrink:0;box-shadow:0 2px 12px rgba(61,111,255,0.4)}
#btn:hover{background:linear-gradient(135deg,#3560e0,#5585ff);transform:scale(1.05)}
.input-hint{text-align:center;margin-top:8px;font-size:.65rem;color:var(--muted)}
</style>
</head>
<body>
<div class="bg-wrap">
  <div class="bg-grad"></div><div class="grid"></div>
  <div class="orb orb1"></div><div class="orb orb2"></div>
</div>
<header>
  <div class="logo-wrap">
    <div class="logo-icon">&#127916;</div>
    <div>
      <div class="logo-text">Cine<span>Agent</span></div>
      <div class="logo-sub">AI Movie Intelligence</div>
    </div>
  </div>
  <div class="badge"><div class="bdot"></div>פעיל</div>
</header>
<div class="main">
  <div id="welcome">
    <div class="wicon">&#127916;</div>
    <div class="wtitle">סינמה אייג&#x27;נט</div>
    <div class="wsub">
      עוזר חכם למציאת סרטים המשלב <b>למידת מכונה</b>, <b>עיבוד שפה טבעית</b> ו<b>ChatGPT AI</b>.<br>
      מסד נתונים של <b>50,000 סרטים</b> עם מידע על פלטפורמות סטרימינג.
    </div>
    <div class="chips">
      <div class="chip" onclick="go('אני רוצה קומדיה רומנטית מצחיקה')"><span class="ctag">חיפוש</span>קומדיה רומנטית</div>
      <div class="chip" onclick="go('סרט אימה עם רוחות ומסתורין')"><span class="ctag">חיפוש</span>אימה ומסתורין</div>
      <div class="chip" onclick="go('movies similar to Inception')"><span class="ctag">דומה</span>דומה ל-Inception</div>
      <div class="chip" onclick="go('סרט אנימציה לילדים עם קסם')"><span class="ctag">חיפוש</span>אנימציה לילדים</div>
      <div class="chip" onclick="go('show me big budget flops')"><span class="ctag">חריגה</span>פלופים</div>
      <div class="chip" onclick="go('find me hidden gems with high rating')"><span class="ctag">חריגה</span>יהלומים נסתרים</div>
      <div class="chip" onclick="go('movies similar to The Dark Knight')"><span class="ctag">דומה</span>דומה ל-Dark Knight</div>
      <div class="chip" onclick="go('what are the movie clusters')"><span class="ctag">קלסטרים</span>הצג קלסטרים</div>
    </div>
  </div>
  <div id="chat-screen">
    <div id="messages"></div>
  </div>
  <div class="input-wrap">
    <div class="input-inner">
      <input id="inp" type="text" placeholder="שאל על סרט, תאר מצב רוח, או בקש המלצה..." autocomplete="off" />
      <button id="btn">&#x2191;</button>
    </div>
    <div class="input-hint">Enter &#x21B5; לשליחה</div>
  </div>
</div>
<script>
var M=document.getElementById('messages'),I=document.getElementById('inp'),B=document.getElementById('btn');
var WEL=document.getElementById('welcome'),CS=document.getElementById('chat-screen'),started=false;
function showChat(){if(!started){started=true;WEL.style.display='none';CS.classList.add('active');}}
function rc(r){return r>=7.5?'#3de0a0':r>=6?'#e8b84b':'#e85050';}
function streamBadges(s){
  if(!s)return '';
  var h='';
  var parts=s.split(', ');
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
  showChat();
  var d=document.createElement('div');d.className='msg '+role;d.innerHTML=html;
  M.appendChild(d);M.scrollTop=M.scrollHeight;
}
function addTyping(){
  showChat();
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
  h+='</div>';return h;
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
