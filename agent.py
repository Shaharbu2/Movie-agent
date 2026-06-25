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

def row_to_result(rank, idx):
    try:
        row = df.loc[int(idx)]
        overview = row["overview"]
        return {
            "rank": rank,
            "title": row["title"],
            "year": int(row["release_year"]) if row["release_year"] else "N/A",
            "genres": row["genres"],
            "rating": round(float(row["vote_average"]), 1),
            "overview": overview[:150] + "..." if len(overview) > 150 else overview,
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
            "stage": "greeting",
            "answers": {},
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
        
        return [row_to_result(i + 1, idx) for i, (idx, row) in enumerate(sorted_df.iterrows())]
    
    except Exception as e:
        print(f"Error in recommend_movies: {e}")
        return []

# ==============================================================
# OPENAI INTEGRATION
# ==============================================================

def call_openai_safe(user_text, stage, answers, results, language):
    """Call OpenAI with error handling."""
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
        if "platform" in answers:
            context_lines.append(f"Platform: {answers['platform']}")
        
        context_str = "\n".join(context_lines) if context_lines else "No preferences yet"
        
        # Build recommendations block
        recs_block = ""
        if results:
            for r in results[:1]:
                recs_block += f"- {r['title']} ({r['year']}), {r['genres']}, ⭐{r['rating']}/10\n"
        
        system_prompt = f"""You are Cinemate, a friendly movie recommendation chatbot.
Your role is to guide users through finding the perfect movie by asking questions one at a time.

Rules:
- Ask ONE question at a time
- Be warm and conversational
- Respond in {language}
- Never invent movies
- Only recommend from dataset results
- Keep responses brief (1-2 sentences)"""
        
        if stage == "ready":
            user_prompt = f"""User preferences: {context_str}
            
Movie to recommend:
{recs_block}

Present this recommendation warmly. Explain briefly why it fits their preferences."""
        else:
            user_prompt = f"""Current conversation stage: {stage}
User said: {user_text}
What we know: {context_str}

Ask the next natural question about their movie preferences. One question only."""
        
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
    
    if stage == "greeting":
        return "היי! 🎬 בואו נמצא סרט מושלם בשבילכם. איזה סגנון בא לכם לראות?" if heb else "Hi! 🎬 Let's find the perfect movie for you. What kind of movie interests you?"
    
    elif stage == "genre":
        return "יופי! ועכשיו - באיזה שנה או עידן אתם רוצים סרט?" if heb else "Great! What era or year do you prefer?"
    
    elif stage == "year":
        return "נחמד! האם יש פלטפורמה מסוימת?" if heb else "Nice! Do you have a streaming platform preference?"
    
    elif stage == "platform":
        return "מעולה! עכשיו בואו נמצא לכם סרט 🎬" if heb else "Perfect! Let me find you a great movie 🎬"
    
    elif stage == "ready":
        if results:
            title = results[0]["title"]
            return f"הנה ההמלצה שלי: {title}. בהנאה לצפייה! 🍿" if heb else f"Here's my recommendation: {title}. Enjoy! 🍿"
        else:
            return "לא מצאתי התאמה טובה. אנא נסו שוב עם העדפות אחרות." if heb else "I couldn't find a good match. Try different preferences."
    
    return "מה הלאה?" if heb else "What next?"

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
        session_id = data.get("session_id", "default")
        
        # Get session
        session = get_or_create_session(session_id)
        stage = session["stage"]
        answers = session["answers"]
        language = "Hebrew" if is_hebrew(user_text) else "English"
        
        # Empty message - just ask next question
        if not user_text:
            q = call_openai_safe(user_text, stage, answers, [], language)
            return jsonify({"reply": q, "results": [], "stage": stage})
        
        # Parse input based on current stage
        if stage == "greeting":
            genres = extract_genres(user_text)
            if genres:
                answers["genre"] = genres[0]  # Take first genre
                session["stage"] = "year"
            else:
                # Ask again
                q = call_openai_safe(user_text, "genre", answers, [], language)
                return jsonify({"reply": q, "results": [], "stage": "genre"})
        
        elif stage == "year":
            year = extract_year(user_text)
            if year:
                answers["year"] = year
                session["stage"] = "platform"
            else:
                # Ask again or move forward
                session["stage"] = "platform"
        
        elif stage == "platform":
            platform = extract_platform(user_text)
            if platform:
                answers["platform"] = platform
            session["stage"] = "ready"
        
        # At ready stage - generate recommendations
        if session["stage"] == "ready":
            results = recommend_movies(answers, top_n=1)
            reply = call_openai_safe(user_text, "ready", answers, results, language)
            session["stage"] = "ready"
            return jsonify({"reply": reply, "results": results, "stage": "ready"})
        
        # Ask next question
        q = call_openai_safe(user_text, session["stage"], answers, [], language)
        return jsonify({"reply": q, "results": [], "stage": session["stage"]})
    
    except Exception as e:
        print(f"ERROR in /chat: {e}")
        print(traceback.format_exc())
        return jsonify({
            "reply": "משהו השתבש. בואו נתחיל מחדש!" if is_hebrew(str(data.get("message", ""))) else "Something went wrong. Let's start over!",
            "results": [],
            "stage": "greeting",
            "error": str(e)
        }), 500

# ==============================================================
# HTML UI
# ==============================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cinemate</title>
<link href="https://fonts.googleapis.com/css2?family=Heebo:wght@400;700;900&display=swap" rel="stylesheet">
<style>
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: 'Heebo', sans-serif;
  background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
  color: #fff;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
header {
  padding: 20px;
  text-align: center;
  background: rgba(0,0,0,0.3);
  border-bottom: 2px solid #e94560;
}
header h1 { margin: 0; font-size: 2.5em; }
header p { margin: 5px 0 0 0; opacity: 0.8; }
.container {
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 20px;
}
.chat-box {
  width: 100%;
  max-width: 600px;
  background: rgba(255,255,255,0.95);
  border-radius: 15px;
  overflow: hidden;
  box-shadow: 0 10px 40px rgba(0,0,0,0.3);
  display: flex;
  flex-direction: column;
  color: #333;
}
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  min-height: 400px;
  max-height: 500px;
}
.msg { margin: 15px 0; display: flex; }
.msg.user { justify-content: flex-end; }
.msg.bot { justify-content: flex-start; }
.bubble {
  max-width: 80%;
  padding: 12px 16px;
  border-radius: 12px;
  line-height: 1.5;
}
.msg.user .bubble { background: #e94560; color: white; }
.msg.bot .bubble { background: #f0f0f0; color: #333; }
.card {
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 15px;
  margin-top: 10px;
  font-size: 0.9em;
}
.card-title { font-weight: 700; color: #e94560; margin: 0 0 5px 0; }
.card-meta { color: #666; font-size: 0.85em; margin: 3px 0; }
.input-area {
  padding: 15px;
  background: #f9f9f9;
  border-top: 1px solid #ddd;
  display: flex;
  gap: 10px;
}
input {
  flex: 1;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 10px;
  font-size: 1em;
  font-family: 'Heebo', sans-serif;
}
button {
  padding: 10px 20px;
  background: #e94560;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 700;
  transition: 0.2s;
}
button:hover { background: #d63450; }
.typing { display: inline-flex; gap: 4px; }
.dot { width: 5px; height: 5px; background: #e94560; border-radius: 50%; animation: bounce 1s infinite; }
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce { 0%, 80%, 100% { opacity: 0.4; } 40% { opacity: 1; } }
</style>
</head>
<body>
<header>
  <h1>🎬 Cinemate</h1>
  <p>Find Your Perfect Movie</p>
</header>
<div class="container">
  <div class="chat-box">
    <div class="chat-messages" id="messages">
      <div class="msg bot">
        <div class="bubble">Hi! 🎬 Let's find the perfect movie for you. What kind of movie interests you today?</div>
      </div>
    </div>
    <div class="input-area">
      <input id="input" placeholder="Type your answer..." autocomplete="off">
      <button id="send">Send</button>
    </div>
  </div>
</div>

<script>
const sessionId = 'sess_' + Math.random().toString(36).substr(2, 9);
const msgDiv = document.getElementById('messages');
const input = document.getElementById('input');

function addMsg(role, text, card = null) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  div.appendChild(bubble);
  
  if (card && role === 'bot') {
    const cardDiv = document.createElement('div');
    cardDiv.className = 'card';
    cardDiv.innerHTML = `
      <div class="card-title">${card.title} (${card.year})</div>
      <div class="card-meta">⭐ ${card.rating}/10 • ${card.genres}</div>
      <div class="card-meta">📺 ${card.streaming}</div>
      <div style="margin-top: 8px; color: #666; font-size: 0.85em;">${card.overview}</div>
    `;
    div.appendChild(cardDiv);
  }
  
  msgDiv.appendChild(div);
  msgDiv.scrollTop = msgDiv.scrollHeight;
}

function addTyping() {
  const div = document.createElement('div');
  div.className = 'msg bot';
  div.id = 'typing';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = '<span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>';
  div.appendChild(bubble);
  msgDiv.appendChild(div);
  msgDiv.scrollTop = msgDiv.scrollHeight;
}

function removeTyping() {
  const typing = document.getElementById('typing');
  if (typing) typing.remove();
}

function send() {
  const text = input.value.trim();
  if (!text) return;
  
  addMsg('user', text);
  input.value = '';
  addTyping();
  
  fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: text, session_id: sessionId })
  })
  .then(r => r.json())
  .then(data => {
    removeTyping();
    addMsg('bot', data.reply || 'No response', data.results ? data.results[0] : null);
  })
  .catch(err => {
    removeTyping();
    addMsg('bot', 'Error: ' + err);
  });
}

document.getElementById('send').onclick = send;
input.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });
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
