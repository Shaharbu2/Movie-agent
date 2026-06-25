import os
import re
import gc
import json
import difflib
import numpy as np
import pandas as pd
from collections import Counter
from flask import Flask, request, jsonify, Response
from datetime import datetime

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
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return text


def normalize_user_text(text):
    return normalize_hebrew_typos(str(text).strip())

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
# 6. GENRE & FILTER MAPPING
# ==============================================================

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

PLATFORM_PATTERNS = {
    "Netflix": ["netflix", "נטפליקס"],
    "Hulu": ["hulu", "הולו"],
    "Prime Video": ["prime", "amazon", "אמזון", "פריים"],
    "Disney+": ["disney", "דיסני", "disney+"],
}

# ==============================================================
# 7. TEXT EXTRACTION HELPERS
# ==============================================================

def extract_genres(text):
    t = text.lower()
    matched = set()
    for kw in sorted(GENRE_KEYWORD_MAP, key=len, reverse=True):
        if kw in t:
            matched.add(GENRE_KEYWORD_MAP[kw])
    return list(matched)

def extract_year(text):
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
    mask = df["title_clean"].apply(lambda tc: bool(tc) and tc in text_clean)
    hits = df[mask]
    if not hits.empty:
        best_idx = hits["title_clean"].str.len().idxmax()
        return df.loc[best_idx, "title"]

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

# ==============================================================
# 8. RECOMMENDATION ENGINE
# ==============================================================

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
    return ", ".join(platforms) if platforms else "לא זמין במאגר"

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

def search_movies(user_preferences, top_n=2):
    """Search for movies based on collected preferences"""
    filtered = df.copy()
    
    genres = user_preferences.get("genres", [])
    year = user_preferences.get("year")
    platform = user_preferences.get("platform")

    if year is not None:
        filtered = filtered[filtered["release_year"] >= year]

    if platform is not None and platform in filtered.columns:
        filtered = filtered[filtered[platform] == 1]

    if genres:
        pattern = "|".join(genres)
        filtered = filtered[filtered["genres"].str.contains(pattern, case=False, na=False)]

    if filtered.empty:
        return []

    # Score by quality
    filtered = filtered.copy()
    filtered["score"] = (
        filtered["vote_average"] * 0.5 +
        (filtered["vote_count"] / filtered["vote_count"].max()) * 50 * 0.3 +
        (filtered["popularity"] / filtered["popularity"].max()) * 50 * 0.2
    )

    top = filtered.nlargest(top_n, "score")
    
    results = [
        row_to_result(i + 1, idx, row["score"])
        for i, (idx, row) in enumerate(top.iterrows())
    ]
    
    return results

# ==============================================================
# 9. CONVERSATION STATE & OPENAI INTEGRATION
# ==============================================================

def build_conversation_context(conv_state):
    """Build context string from conversation state"""
    parts = []
    if conv_state.get("genres"):
        parts.append(f"ז׳אנרים: {', '.join(conv_state['genres'])}")
    if conv_state.get("year"):
        parts.append(f"שנה: מ-{conv_state['year']} ומעלה")
    if conv_state.get("platform"):
        parts.append(f"פלטפורמה: {conv_state['platform']}")
    return " | ".join(parts) if parts else "עדיין לא נאספו העדפות"

def call_openai(user_text, conv_state, results=None):
    """Call OpenAI for conversational response."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return fallback_reply(user_text, conv_state, results)

    try:
        import urllib.request

        language = "Hebrew" if is_hebrew(user_text) else "English"
        state = conv_state.get("state", "greeting")
        questions_asked = conv_state.get("questions_asked", 0)
        
        # Build data block - ONLY from database results
        data_block = ""
        if results:
            for r in results[:2]:
                data_block += (
                    f"- {r['title']} ({r['year']}), ז׳אנרים: {r['genres']}, "
                    f"דירוג: {r['rating']}/10, זמין ב: {r['streaming']}\n"
                )
        
        collected = build_conversation_context(conv_state)

        system_prompt = """
You are MovieMate, a friendly and conversational movie recommendation AI.

YOUR JOB: Ask exactly 3 questions to understand the user's movie preferences, then recommend movies.

CRITICAL RULES:
1. You ONLY recommend movies from the provided database results.
2. You NEVER invent, imagine, or suggest movies that are not in the results.
3. If no results exist, say so honestly.
4. Ask ONE question at a time. Wait for response.
5. Never ask the same question twice.
6. Keep responses SHORT (1-2 sentences max).

THE 3 QUESTIONS (in this order):
1. "מה הסגנון המועדף עליך? קומדיה, אקשן, דרמה, אימה, רומנטיקה, אנימציה או משהו אחר?"
2. "בחרת סגנון טוב! עכשיו, מאיזה שנה בערך? (לדוגמא: 2020 ומעלה, או 2015-2023)"
3. "מעולה! עכשיו, תוכל לתת דוגמא לסרט בסגנון הזה שאהבת?"

After 3 answers: Give recommendations ONLY from database results.

If user asks something off-topic: "אני כאן בשביל להמליץ על סרטים 🎬" and redirect.

Language: Hebrew (עברית) for Hebrew users, English for English users.
"""

        user_prompt = (
            f"User language: {language}\n"
            f"Conversation state: {state}\n"
            f"User message: {user_text}\n"
            f"Questions already asked: {questions_asked}\n"
            f"Collected preferences: {collected}\n\n"
        )
        
        if state == "greeting":
            user_prompt += (
                "This is the FIRST message. The user just opened the chat.\n"
                "Greet them warmly in Hebrew and say you'll ask 3 focused questions to find them the perfect movie.\n"
                "Then ask Question 1: What's their favorite genre?"
            )
        elif state == "gathering":
            user_prompt += (
                f"You've already asked {questions_asked} questions.\n"
                f"Current info: {collected}\n"
                f"Ask Question {questions_asked + 1} next.\n"
                "Be clear and direct. Ask only ONE question. Wait for their answer."
            )
        elif state == "ready":
            user_prompt += (
                f"You have enough info to recommend movies:\n{collected}\n"
                f"Database results to present:\n{data_block if data_block else 'No movies found in database with these criteria.'}\n"
                "Present ONLY these database results. Never invent movies. If no results, say honestly."
            )
        elif state == "out_of_scope":
            user_prompt += (
                "The user asked something not related to movie recommendations.\n"
                "Politely but firmly explain you're focused on movie recommendations only and redirect them back."
            )

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 150,
            "temperature": 0.5
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
        return fallback_reply(user_text, conv_state, results)

def fallback_reply(user_text, conv_state, results=None):
    heb = is_hebrew(user_text)
    state = conv_state.get("state", "greeting")
    
    if state == "greeting":
        return "היי! 👋 אני כאן כדי לעזור לך למצוא סרט מדהים 🎬\nבואו נתחיל - איזה סוג סרט אתה אוהב?" if heb else "Hey! 👋 I'm here to help you find an amazing movie 🎬\nLet's start - what kind of movies do you enjoy?"
    
    if state == "out_of_scope":
        return "אני כאן בשביל להמליץ על סרטים 🎬 בואו נחזור לשיחה על סרטים!" if heb else "I'm here to help with movie recommendations 🎬. Let's get back to finding you a great movie!"
    
    if state == "ready" and results:
        titles = ", ".join([r["title"] for r in results[:2]])
        return f"הנה ההמלצות שלי: {titles}" if heb else f"Here are my picks: {titles}"
    
    return "בואו נמשיך לדבר על סרטים 🎬" if heb else "Let's keep talking about movies 🎬"

# ==============================================================
# 10. ROUTES
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
    conv_state = data.get("state", {
        "state": "greeting",
        "genres": [],
        "year": None,
        "platform": None,
        "questions_asked": 0
    })

    if not user_text:
        return jsonify({
            "reply": "?",
            "state": conv_state,
            "results": []
        })

    # Extract any new preferences from user message
    new_genres = extract_genres(user_text)
    new_year = extract_year(user_text)
    new_platform = extract_platform(user_text)

    # Update state with new info
    if new_genres:
        for g in new_genres:
            if g not in conv_state["genres"]:
                conv_state["genres"].append(g)
    
    if new_year and not conv_state["year"]:
        conv_state["year"] = new_year
    
    if new_platform and not conv_state["platform"]:
        conv_state["platform"] = new_platform

    # Check if this is out of scope
    is_off_topic = False
    if conv_state["state"] == "gathering":
        # In gathering, only mark as off-topic if it's clearly not about movies
        is_off_topic = not any([
            new_genres, new_year, new_platform,
            len(clean_text(user_text)) < 30,  # Short responses are assumed to be answering questions
            any(w in clean_text(user_text) for w in ["סרט", "movie", "film", "netflix", "nflx", "prime", "disney", "הולו"])
        ])
    else:
        is_off_topic = not any([
            new_genres, new_year, new_platform,
            any(w in clean_text(user_text) for w in ["סרט", "movie", "film", "watch", "recommend", "like", "כמו", "אהבתי"]),
            conv_state["state"] == "greeting"
        ])

    # Determine next state
    if conv_state["state"] == "greeting":
        conv_state["state"] = "gathering"
        conv_state["questions_asked"] = 1
    elif conv_state["state"] == "gathering":
        conv_state["questions_asked"] += 1
        # Only go to ready after exactly 3 questions AND got some preference info
        if conv_state["questions_asked"] >= 3 and (conv_state["genres"] or conv_state["year"] or conv_state["platform"]):
            conv_state["state"] = "ready"

    if is_off_topic and conv_state["state"] != "greeting":
        conv_state["state"] = "out_of_scope"
    elif conv_state["state"] == "out_of_scope" and (new_genres or new_year or new_platform):
        conv_state["state"] = "gathering"

    # Get recommendations if ready
    results = []
    if conv_state["state"] == "ready":
        results = search_movies({
            "genres": conv_state["genres"],
            "year": conv_state["year"],
            "platform": conv_state["platform"]
        }, top_n=2)

    # Get OpenAI response
    reply = call_openai(user_text, conv_state, results)

    return jsonify({
        "reply": reply,
        "state": conv_state,
        "results": results
    })

# ==============================================================
# 11. HTML PAGE
# ==============================================================

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>cinemate</title>
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
.stage {{
  position:relative;
  z-index:2;
  width:min(900px, 92vw);
  margin:40px auto 34px;
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
  height:480px;
  overflow-y:auto;
  padding:20px;
  border:5px solid rgba(215,25,32,.18);
  margin-bottom:14px;
}}
.msg {{ display:flex; margin:12px 0; gap:8px; }}
.msg.user {{ justify-content:flex-start; }}
.msg.bot {{ justify-content:flex-end; }}
.msg.bot.has-cards {{ flex-direction:column; align-items:flex-end; }}
.bubble {{
  max-width:70%;
  padding:13px 16px;
  border-radius:20px;
  line-height:1.65;
  font-size:16px;
  box-shadow:0 6px 16px rgba(0,0,0,.08);
  white-space:pre-line;
  word-wrap:break-word;
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
.input-row {{ display:flex; gap:10px; }}
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
  transition:.2s;
}}
#btn:hover {{ transform:translateY(-2px); box-shadow:0 14px 28px rgba(215,25,32,.5); }}
#btn:active {{ transform:translateY(0); }}
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
  .stage {{ width:94vw; margin:30px auto; }}
  .content {{ padding:15px; }}
  .chat {{ height:420px; }}
  .bubble,.cards {{ max-width:90%; }}
  .input-row {{ flex-direction:column; }}
  #btn {{ padding:13px; }}
}}
</style>
</head>
<body>
<div class="marquee"></div>
<header>
  <div class="logo">🎬 <span>cinemate</span></div>
</header>

<main class="stage">
  <div class="stage-top">NOW SHOWING • MOVIE AGENT • NOW SHOWING</div>
  <div class="content">
    <div class="quick-title">בחר דוגמה או שאל משהו:</div>
    <div class="chips">
      <button class="chip" onclick="go('קומדיה משנת 2000')">קומדיה משנת 2000</button>
      <button class="chip" onclick="go('סרטי אקשן שקיימים בנטפליקס')">אקשן בנטפליקס</button>
      <button class="chip" onclick="go('סרטים דומים ל-Inception')">דומה ל-Inception</button>
      <button class="chip" onclick="go('יהלומים נסתרים')">יהלומים נסתרים</button>
    </div>

    <div id="chat" class="chat"></div>

    <div class="input-row">
      <input id="inp" placeholder="כתוב לי מה בא לך לראות..." autocomplete="off">
      <button id="btn">שליחה</button>
    </div>
  </div>
</main>

<script>
const chat = document.getElementById('chat');
const inp = document.getElementById('inp');
const btn = document.getElementById('btn');

let convState = {{
  state: 'greeting',
  genres: [],
  year: null,
  platform: null,
  questions_asked: 0
}};

function esc(s) {{
  return String(s || '').replace(/[&<>"']/g, m => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m]));
}}

function add(role, html) {{
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.innerHTML = html;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}}

function addTyping() {{
  add('bot', '<div class="bubble" id="typing"><span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span></div>');
}}

function rmTyping() {{
  const t = document.getElementById('typing');
  if(t) t.parentElement.remove();
}}

function cards(results) {{
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

function send() {{
  const text = inp.value.trim();
  if(!text) return;
  
  add('user', '<div class="bubble">' + esc(text) + '</div>');
  inp.value = '';
  addTyping();

  fetch('/chat', {{
    method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{
      message: text,
      state: convState
    }})
  }})
  .then(r => r.json())
  .then(data => {{
    rmTyping();
    convState = data.state;
    const hasResults = data.results && data.results.length > 0;
    const cleanReply = (data.reply || '').replace(/\*\*([^*]+)\*\*/g, '$1').replace(/\*([^*]+)\*/g, '$1');
    let extra = '';
    if(hasResults) {{
      extra = cards(data.results);
    }}
    const msgClass = hasResults ? 'bot has-cards' : 'bot';
    add(msgClass, '<div class="bubble">' + esc(cleanReply) + '</div>' + extra);
  }})
  .catch(err => {{
    rmTyping();
    add('bot', '<div class="bubble">אופס, משהו השתבש 😅</div>');
  }});
}}

btn.onclick = send;
inp.addEventListener('keydown', e => {{
  if(e.key === 'Enter') send();
}});

function go(q) {{
  inp.value = q;
  send();
}}

// Send initial greeting
setTimeout(() => {{
  fetch('/chat', {{
    method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{
      message: 'היי',
      state: convState
    }})
  }})
  .then(r => r.json())
  .then(data => {{
    convState = data.state;
    add('bot', '<div class="bubble">' + esc(data.reply || 'שלום!') + '</div>');
  }});
}}, 300);
</script>
</body>
</html>"""

# ==============================================================
# 12. RUN
# ==============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
