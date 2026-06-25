import os

import re

import gc

import json

import difflib

import numpy as np

import pandas as pd

from collections import Counter, defaultdict

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

# 0. SESSION STATE TRACKING (for interview flow)

# ==============================================================

SESSIONS = {}  # key: session_id, value: {stage, answers_collected, timestamp}



def get_session(sid):

    if sid not in SESSIONS:

        SESSIONS[sid] = {

            "stage": "greeting",

            "answers": {},

            "timestamp": datetime.now()

        }

    return SESSIONS[sid]



def advance_stage(session):

    """Move to next interview stage based on answers collected."""

    collected = set(session["answers"].keys())

    stages = ["greeting", "genre", "year", "reference", "context", "platform", "ready"]

    

    if "genre" not in collected:

        session["stage"] = "genre"

    elif "year" not in collected:

        session["stage"] = "year"

    elif "reference" not in collected:

        session["stage"] = "reference"

    elif "context" not in collected:

        session["stage"] = "context"

    elif "platform" not in collected:

        session["stage"] = "platform"

    else:

        session["stage"] = "ready"



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



# ==============================================================

# 7. RECOMMENDATION LOGIC

# ==============================================================



def recommend_from_answers(answers, top_n=3):

    """Generate recommendations based on collected interview answers."""

    filtered = df.copy()

    

    # Apply genre filter if provided

    if "genre" in answers and answers["genre"]:

        genres = answers["genre"]

        if isinstance(genres, str):

            genres = [genres]

        pattern = "|".join(genres)

        filtered = filtered[filtered["genres"].str.contains(pattern, case=False, na=False)]

    

    # Apply year filter if provided

    if "year" in answers and answers["year"]:

        year = answers["year"]

        filtered = filtered[filtered["release_year"] >= year]

    

    # Apply platform filter if provided

    if "platform" in answers and answers["platform"]:

        platform = answers["platform"]

        if platform in filtered.columns:

            filtered = filtered[filtered[platform] == 1]

    

    if filtered.empty:

        return []

    

    # If reference movie provided, use similarity

    if "reference" in answers and answers["reference"]:

        movie_title = answers["reference"]

        matches = df[df["title"].str.lower() == str(movie_title).lower()]

        if matches.empty:

            matches = df[df["title"].str.lower().str.contains(str(movie_title).lower(), na=False)]

        

        if not matches.empty:

            idx = int(matches.index[0])

            scores = cosine_similarity(sim_data_sparse[idx:idx+1], sim_data_sparse).flatten()

            filtered["sim_score"] = scores

            filtered = filtered[filtered.index != idx]

            filtered = filtered[filtered["sim_score"] >= 0.05]

            top_results = filtered.sort_values("sim_score", ascending=False).head(top_n)

            return [row_to_result(i + 1, tidx, row["sim_score"]) 

                   for i, (tidx, row) in enumerate(top_results.iterrows())]

    

    # Default: rank by rating, popularity, vote count

    top_results = filtered.sort_values(["vote_average", "vote_count", "popularity"], ascending=False).head(top_n)

    return [row_to_result(i + 1, idx, row["vote_average"] / 10) 

           for i, (idx, row) in enumerate(top_results.iterrows())]



# ==============================================================

# 8. OPENAI RESPONSE - interview flow

# ==============================================================



def call_openai(user_text, stage, answers, results, language):

    """Call OpenAI for conversational next question or final recommendations."""

    api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:

        return fallback_reply(stage, answers, results, language)

    

    try:

        import urllib.request

        

        # Build context about what we know

        context_lines = []

        if "genre" in answers:

            context_lines.append(f"User prefers: {answers['genre']}")

        if "year" in answers:

            context_lines.append(f"User wants movies from: {answers['year']} onwards")

        if "reference" in answers:

            context_lines.append(f"Reference movie they liked: {answers['reference']}")

        if "context" in answers:

            context_lines.append(f"Viewing context: {answers['context']}")

        if "platform" in answers:

            context_lines.append(f"Preferred platform: {answers['platform']}")

        

        context_str = "\n".join(context_lines) if context_lines else "No preferences collected yet."

        

        # Build recommendations block for final stage

        recommendations_block = ""

        if stage == "ready" and results:

            for r in results[:3]:

                streaming = r.get("streaming") or "Not available in dataset"

                recommendations_block += (

                    f"- {r['title']} ({r['year']}), genres: {r['genres']}, "

                    f"rating: {r['rating']}/10, streaming: {streaming}\n"

                )

        

        system_prompt = """

You are Cinemate, a friendly and enthusiastic movie recommendation chatbot.

Your role is to guide users through a personalized movie discovery experience by asking smart, conversational questions one at a time.

Important rules:

- Speak warmly and naturally — you're having a conversation, not running a survey.

- Ask ONE question at a time and wait for the user's answer before proceeding to the next.

- Never skip stages or ask multiple questions in one message.

- Match the user's language: Hebrew for Hebrew input, English for English input.

- You are ONLY allowed to help with movie recommendations. If users ask unrelated questions, politely redirect them.

- When stage is "ready": Present the recommendations confidently based on dataset results. Do NOT invent movies.

- Do NOT use markdown formatting (**bold** or *italic*) — write plain text only.

- Keep responses concise: 1-2 sentences for interview questions, 2-3 for final recommendations.

Interview flow stages:

1. "genre": Ask what genre or style they're interested in (action, comedy, drama, etc.)

2. "year": Ask what year range they prefer (recent, classics, specific year, etc.)

3. "reference": Ask if they have a favorite movie as a reference (to find similar films).

4. "context": Ask about the viewing occasion (date night, solo watch, with friends, etc.).

5. "platform": Ask their streaming platform preference (Netflix, Hulu, Prime Video, Disney+).

6. "ready": Present 3 personalized recommendations with a brief explanation of why they fit.

"""

        

        if stage == "ready":

            user_prompt = (

                f"Answer language: {language}\n"

                f"User's final message: {user_text}\n"

                f"User preferences collected:\n{context_str}\n\n"

                f"Recommended movies:\n{recommendations_block}\n\n"

                f"Write a friendly, concise message presenting these {len(results)} recommendations. "

                f"Explain briefly why they match the user's preferences. "

                f"Do NOT list movie titles as a numbered list — weave them into natural sentences."

            )

        else:

            user_prompt = (

                f"Answer language: {language}\n"

                f"Current stage: {stage}\n"

                f"User message: {user_text}\n"

                f"What we know so far:\n{context_str}\n\n"

                f"Ask the next interview question naturally. Keep it conversational and brief (1-2 sentences)."

            )

        

        payload = {

            "model": "gpt-4o-mini",

            "messages": [

                {"role": "system", "content": system_prompt},

                {"role": "user", "content": user_prompt}

            ],

            "max_tokens": 260,

            "temperature": 0.45

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

        return fallback_reply(stage, answers, results, language)



def fallback_reply(stage, answers, results, language):

    """Fallback responses when OpenAI is unavailable."""

    heb = language == "Hebrew"

    

    if stage == "greeting":

        return "שלום! בואו נמצא ביחד את הסרט המושלם בשבילכם 🎬" if heb else "Hi! Let's find the perfect movie for you together 🎬"

    elif stage == "genre":

        return "מה סוג הסרט שמעניין אתכם? (אקשן, קומדיה, דרמה וכו')" if heb else "What kind of movie interests you? (action, comedy, drama, etc.)"

    elif stage == "year":

        return "באיזה שנים אתם רוצים שהסרט יהיה? (סרט חדש, קלאסיקה, שנה מסוימת?)" if heb else "What year range do you prefer? (recent, classic, specific year?)"

    elif stage == "reference":

        return "האם יש לכם סרט מועדף שאפשר להשתמש בו כהשוואה?" if heb else "Do you have a favorite movie we can use as a reference?"

    elif stage == "context":

        return "מה ההזדמנות? (לילה רומנטי, צפייה בודדת, עם חברים?)" if heb else "What's the occasion? (romantic night, solo watch, with friends?)"

    elif stage == "platform":

        return "איזה פלטפורמת סטרימינג קיימת אצלכם?" if heb else "Which streaming platform do you have?"

    elif stage == "ready":

        if not results:

            return "לא מצאתי התאמה מושלמת בדאטה שלנו. אנא נסו לשנות כמה עדיפויות." if heb else "I couldn't find a perfect match. Please adjust your preferences."

        

        titles = ", ".join([r["title"] for r in results[:3]])

        return f"הנה ההמלצות שלי: {titles}. אני חושב שזה יתאים לכם!" if heb else f"Here are my recommendations: {titles}. I think you'll love them!"

    

    return "משהו השתבש. אנא נסו שוב." if heb else "Something went wrong. Please try again."



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

    session_id = data.get("session_id", "default")

    

    if not user_text:

        return jsonify({"reply": "", "results": [], "stage": "greeting"})

    

    # Get or create session

    session = get_session(session_id)

    stage = session["stage"]

    answers = session["answers"]

    language = "Hebrew" if is_hebrew(user_text) else "English"

    

    # Parse user input to extract relevant data for current stage

    if stage == "genre":

        genres = extract_genres(user_text)

        if genres:

            answers["genre"] = genres

    elif stage == "year":

        year = extract_year(user_text)

        if year:

            answers["year"] = year

    elif stage == "reference":

        movie = find_movie_title(user_text)

        if movie:

            answers["reference"] = movie

    elif stage == "context":

        # Just store the user's text as context (viewing occasion)

        if user_text and len(user_text) > 3:

            answers["context"] = user_text

    elif stage == "platform":

        platform = extract_platform(user_text)

        if platform:

            answers["platform"] = platform

    

    # Generate recommendations if we're at ready stage

    results = []

    if stage == "ready":

        results = recommend_from_answers(answers, top_n=3)

    

    # Advance to next stage

    advance_stage(session)

    

    # Get OpenAI response

    reply = call_openai(user_text, session["stage"], answers, results, language)

    

    return jsonify({

        "reply": reply,

        "results": results,

        "stage": session["stage"],

        "session_id": session_id

    })



# ==============================================================

# 10. HTML - Cinemate Design

# ==============================================================



HTML_PAGE = f"""<!DOCTYPE html>

<html lang="he" dir="rtl">

<head>

<meta charset="UTF-8">

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Cinemate</title>

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

  padding:8px 18px 24px;

}}

.hero h1 {{

  margin:0;

  font-size:clamp(42px, 8vw, 72px);

  line-height:1.1;

  font-weight:900;

  text-shadow:0 6px 0 rgba(215,25,32,.45), 0 0 28px rgba(255,48,64,.24);

  letter-spacing:-1px;

}}

.stage {{

  position:relative;

  z-index:2;

  width:min(1120px, 92vw);

  margin:0 auto 34px;

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

  font-size:14px;

}}

.content {{ padding:24px; }}

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

  .hero {{ padding:8px 18px 18px; }}

  .hero h1 {{ font-size:clamp(32px, 7vw, 52px); }}

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

  <div class="logo">🎬 Cinemate</div>

  <div class="badge">AI • {len(df):,} films</div>

</header>



<section class="hero">

  <h1>Find Your Next Favorite Film</h1>

</section>



<main class="stage">

  <div class="stage-top">CINEMATE • GUIDED DISCOVERY • CINEMATE</div>

  <div class="content">

    <div id="chat" class="chat">

      <div class="msg bot">

        <div class="bubble">Hi! Let's find the perfect movie for you today. I'll ask you a few quick questions to narrow it down. Ready? 🍿</div>

      </div>

    </div>



    <div class="input-row">

      <input id="inp" placeholder="Tell me what you're in the mood for..." autocomplete="off">

      <button id="btn">Send</button>

    </div>

  </div>

</main>



<script>

const chat = document.getElementById('chat');

const inp = document.getElementById('inp');

const btn = document.getElementById('btn');

const sessionId = 'session_' + Math.random().toString(36).substr(2, 9);



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

      <div class="meta">${{esc(r.year)}} • ⭐ ${{esc(r.rating)}}/10</div>

      <div class="genres">${{esc(r.genres)}}</div>

      ${{r.streaming ? `<div class="stream">Available on: ${{esc(r.streaming)}}</div>` : ''}}

      <div class="desc">${{esc(r.overview)}}</div>

    </div>`;

  }});

  h += '</div>';

  return h;

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

    body:JSON.stringify({{message:text, session_id:sessionId}})

  }})

  .then(r => r.json())

  .then(data => {{

    rmTyping();

    const hasResults = data.results && data.results.length > 0;

    const cleanReply = (data.reply || '').replace(/\*\*([^*]+)\*\*/g, '$1').replace(/\*([^*]+)\*/g, '$1');

    let extra = '';

    if (hasResults) {{

      extra = cards(data.results);

    }}

    const msgClass = hasResults ? 'bot has-cards' : 'bot';

    add(msgClass, '<div class="bubble">' + esc(cleanReply) + '</div>' + extra);

  }})

  .catch(() => {{

    rmTyping();

    add('bot', '<div class="bubble">Something went wrong. Please try again.</div>');

  }});

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
