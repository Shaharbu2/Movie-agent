import os
import re
import gc
import json
import numpy as np
import pandas as pd
from collections import Counter
from flask import Flask, request, jsonify, Response
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
import traceback

app = Flask(__name__)

# ==============================================================
# CONFIGURATION
# ==============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "movies_master.csv")

# Session storage
SESSIONS = {}

# ==============================================================
# LOAD DATA
# ==============================================================

print("Loading data...")
try:
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
            else:
                df[col] = ""
    
    df["title"] = df["title"].fillna("").astype(str)
    df["overview"] = df["overview"].fillna("").astype(str)
    df["genres"] = df["genres"].fillna("").astype(str)
    
    for c in ["vote_average", "popularity", "runtime", "vote_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.float32)
    
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(np.int16)
    
    for col in ["Netflix", "Hulu", "Prime Video", "Disney+"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int8)
    
    print(f"✓ Loaded {len(df)} movies")
    
except Exception as e:
    print(f"ERROR loading data: {e}")
    df = pd.DataFrame()

# ==============================================================
# TEXT PROCESSING
# ==============================================================

def is_hebrew(text):
    return bool(re.search(r"[\u0590-\u05FF]", str(text)))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\u0590-\u05FF\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# Genre mapping
GENRE_KEYWORD_MAP = {
    "action": "Action", "אקשן": "Action",
    "comedy": "Comedy", "קומדיה": "Comedy",
    "drama": "Drama", "דרמה": "Drama",
    "horror": "Horror", "אימה": "Horror",
    "romance": "Romance", "רומנטי": "Romance",
    "thriller": "Thriller", "מתח": "Thriller",
    "animation": "Animation", "אנימציה": "Animation",
    "adventure": "Adventure", "הרפתקה": "Adventure",
    "fantasy": "Fantasy", "פנטזיה": "Fantasy",
    "mystery": "Mystery", "מסתורין": "Mystery",
}

PLATFORM_PATTERNS = {
    "Netflix": ["netflix", "נטפליקס"],
    "Hulu": ["hulu", "הולו"],
    "Prime Video": ["prime", "amazon", "אמזון"],
    "Disney+": ["disney", "דיסני"],
}

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
    return ", ".join(platforms) if platforms else "Not available"

def row_to_result(rank, idx, score=0):
    try:
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
    except:
        return None

# ==============================================================
# SESSION MANAGEMENT
# ==============================================================

def get_or_create_session(session_id):
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "stage": "genre",  # Start with genre question
            "answers": {},
            "done": False,
        }
    return SESSIONS[session_id]

# ==============================================================
# RECOMMENDATION ENGINE
# ==============================================================

def recommend_movies(answers, top_n=1):
    """Generate recommendations based on collected answers."""
    try:
        filtered = df.copy()
        
        # Apply filters
        if "genre" in answers and answers["genre"]:
            genres = answers["genre"]
            if isinstance(genres, str):
                genres = [genres]
            pattern = "|".join(genres)
            filtered = filtered[filtered["genres"].str.contains(pattern, case=False, na=False)]
        
        if "year" in answers and answers["year"]:
            year = answers["year"]
            filtered = filtered[filtered["release_year"] >= year]
        
        if "platform" in answers and answers["platform"]:
            platform = answers["platform"]
            if platform in filtered.columns:
                filtered = filtered[filtered[platform] == 1]
        
        if filtered.empty:
            return []
        
        # Sort by rating and popularity
        sorted_df = filtered.sort_values(
            ["vote_average", "vote_count", "popularity"], 
            ascending=False
        ).head(top_n)
        
        return [row_to_result(i + 1, idx, row["vote_average"] / 10.0) 
                for i, (idx, row) in enumerate(sorted_df.iterrows())]
    
    except Exception as e:
        print(f"Error in recommend_movies: {e}")
        return []

# ==============================================================
# OPENAI INTEGRATION
# ==============================================================

def call_openai_safe(user_text, stage, answers, results, language, is_post_recommendation=False):
    """Call OpenAI only for recommendations and post-recommendation chat."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    
    if not api_key:
        return get_fallback_reply(stage, answers, results, language)
    
    try:
        import urllib.request
        
        # Build context
        context_lines = []
        if "genre" in answers:
            context_lines.append(f"Genre: {answers['genre']}")
        if "year" in answers:
            context_lines.append(f"Year from: {answers['year']}")
        if "occasion" in answers:
            context_lines.append(f"Occasion: {answers['occasion']}")
        if "reference" in answers:
            context_lines.append(f"Reference movie: {answers['reference']}")
        
        context_str = "\n".join(context_lines) if context_lines else "No preferences yet"
        
        # Build recommendations block
        recs_block = ""
        if results:
            for r in results[:1]:
                recs_block += f"- {r['title']} ({r['year']}), {r['genres']}, ⭐{r['rating']}/10\n"
        
        system_prompt = f"""You are Cinemate, a friendly movie recommendation chatbot.
You help users find movies through natural conversation.

Rules:
- Be warm and conversational
- Respond in {language}
- Never invent movies
- Only recommend from dataset results
- Keep responses brief and natural
- After recommending a movie, you can continue having normal conversations
- Only give new recommendations if the user specifically asks for another movie"""
        
        if stage == "ready":
            user_prompt = f"""User preferences: {context_str}
            
Movie to recommend:
{recs_block}

Present this recommendation warmly. Explain briefly why it fits their preferences. Ask if they want to chat about something else or get another recommendation."""
        elif is_post_recommendation:
            user_prompt = f"""You already recommended a movie to this user.
Now they are asking: {user_text}

Just answer their question naturally as a helpful assistant. Don't try to recommend another movie unless they specifically ask for one."""
        else:
            # Don't use OpenAI for interview questions - use simple logic instead
            return get_next_question(stage, answers, language)
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.6
        }
        
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=20) as response:
            data = json.loads(response.read())
            return data["choices"][0]["message"]["content"]
    
    except Exception as e:
        print(f"OpenAI error: {e}")
        print(traceback.format_exc())
        return get_fallback_reply(stage, answers, results, language)

def get_fallback_reply(stage, answers, results, language):
    """Fallback responses when OpenAI is unavailable."""
    heb = language == "Hebrew"
    
    if stage == "genre":
        return "איזה ז'אנר בא לכם? (דרמה, קומדיה, אקשן וכו')" if heb else "What genre? (drama, comedy, action, etc.)"
    
    elif stage == "year":
        return "מאיזה שנה? (למשל 2020 או 'אחרון')" if heb else "What year? (e.g., 2020 or 'recent')"
    
    elif stage == "occasion":
        return "לאיזה הזדמנות? (דייט, חברים, משפחה וכו')" if heb else "What's the occasion? (date, friends, family, etc.)"
    
    elif stage == "reference":
        return "יש סרט דומה שאתה אוהב?" if heb else "Is there a similar movie you love?"
    
    elif stage == "ready":
        if results:
            title = results[0]["title"]
            return f"הנה ההמלצה שלי: {title}. בהנאה! 🍿" if heb else f"Here's my recommendation: {title}. Enjoy! 🍿"
        else:
            return "לא מצאתי התאמה טובה. אנא נסו שוב עם העדפות אחרות." if heb else "I couldn't find a good match. Try different preferences."
    
    return ""

# ==============================================================
# ROUTES
# ==============================================================

@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "movies": int(len(df)),
        "openai_key_set": bool(os.environ.get("OPENAI_API_KEY", "").strip())
    })

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        user_text = str(data.get("message", "")).strip()
        session_id = request.headers.get('X-Session-Id', 'default')
        
        # Get session
        session = get_or_create_session(session_id)
        stage = session["stage"]
        answers = session["answers"]
        language = "Hebrew" if is_hebrew(user_text) else "English"
        
        # Check for reset command
        if user_text.lower() in ["סרט חדש", "new movie", "מחדש", "reset", "סרט אחר"]:
            SESSIONS[session_id] = {
                "stage": "genre",
                "answers": {},
                "done": False,
            }
            session = SESSIONS[session_id]
            q = "בואו נמצא סרט מושלם! איזה ז'אנר או סגנון בא לכם לראות?" if language == "Hebrew" else "Let's find a perfect movie! What genre or style are you in the mood for?"
            return jsonify({"reply": q, "results": [], "stage": "genre", "reset": True})
        
        # Empty message - just ask next question
        if not user_text:
            q = get_next_question(stage, answers, language)
            return jsonify({"reply": q, "results": [], "stage": stage})
        
        # If we've already given a recommendation and user is asking something else, just answer naturally
        if session.get("done"):
            reply = call_openai_safe(user_text, "post_recommendation", answers, [], language, is_post_recommendation=True)
            return jsonify({"reply": reply, "results": [], "stage": "done"})
        
        # Parse input based on current stage
        if stage == "genre":
            genres = extract_genres(user_text)
            if genres:
                answers["genre"] = genres[0]  # Take first genre
            # Move to next stage regardless of whether we found a genre
            session["stage"] = "year"
            q = get_next_question("year", answers, language)
            return jsonify({"reply": q, "results": [], "stage": "year"})
        
        elif stage == "year":
            year = extract_year(user_text)
            if year:
                answers["year"] = year
            # Move to next stage regardless
            session["stage"] = "occasion"
            q = get_next_question("occasion", answers, language)
            return jsonify({"reply": q, "results": [], "stage": "occasion"})
        
        elif stage == "occasion":
            # Store occasion/context
            if user_text and len(user_text) > 2:
                answers["occasion"] = user_text
            # Move to reference movie stage
            session["stage"] = "reference"
            q = get_next_question("reference", answers, language)
            return jsonify({"reply": q, "results": [], "stage": "reference"})
        
        elif stage == "reference":
            # Try to find reference movie
            movie_title = find_movie_title(user_text)
            if movie_title:
                answers["reference"] = movie_title
            # Ready to recommend - don't ask more questions
            session["stage"] = "ready"
        
        # At ready stage - generate recommendations
        if session["stage"] == "ready":
            results = recommend_movies(answers, top_n=1)
            reply = call_openai_safe(user_text, "ready", answers, results, language)
            session["stage"] = "ready"
            session["done"] = True  # Mark as done - user can now ask other questions
            return jsonify({"reply": reply, "results": results, "stage": "ready"})
        
        # Fallback
        q = get_next_question(session["stage"], answers, language)
        return jsonify({"reply": q, "results": [], "stage": session["stage"]})
    
    except Exception as e:
        print(f"ERROR in /chat: {e}")
        print(traceback.format_exc())
        return jsonify({
            "reply": "משהו השתבש. נסו שוב בעוד רגע." if is_hebrew(str(data.get("message", ""))) else "Something went wrong. Please try again.",
            "results": [],
            "stage": "genre",
            "error": str(e)
        }), 500

def get_next_question(stage, answers, language):
    """Get the next question based on current stage."""
    heb = language == "Hebrew"
    
    if stage == "genre":
        return "איזה ז'אנר או סגנון בא לכם לראות? (דרמה, קומדיה, אקשן, רומנטיקה וכו')" if heb else "What genre or style would you like? (drama, comedy, action, romance, etc.)"
    
    elif stage == "year":
        return "מאיזה שנה הסרט? (למשל 2020, או שנות ה-2010, או 'אחרון')" if heb else "What year? (e.g., 2020, 2010s, or 'recent')"
    
    elif stage == "occasion":
        return "לאיזה הזדמנות או אירוע? (דייט, ערב עם חברים, משפחה, ערב בודד וכו')" if heb else "What's the occasion? (date night, with friends, family, solo night, etc.)"
    
    elif stage == "reference":
        return "יש לך סרט שאהבת שמתאים למה שאתה מחפש? (אם אין בסדר - כתוב 'לא')" if heb else "Is there a movie you loved that's similar to what you're looking for? (or write 'no')"
    
    return ""

# ==============================================================
# HTML UI - ORIGINAL DESIGN PRESERVED EXACTLY
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

        <div class="bubble">היי, ברוכים הבאים ל-Cinemate 🎬 בואו נמצא יחד סרט שמתאים למצב הרוח שלכם. איזה ז'אנר או סגנון בא לכם לראות?</div>

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

const sessionId = 'sess_' + Math.random().toString(36).substr(2, 16);



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

    add('bot', '<div class="bubble">משהו השתבש. נסו שוב בעוד רגע.</div>');

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
# RUN
# ==============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Cinemate on port {port}...")
    print(f"OpenAI API key: {'SET' if os.environ.get('OPENAI_API_KEY', '').strip() else 'NOT SET'}")
    app.run(host="0.0.0.0", port=port, debug=False)
