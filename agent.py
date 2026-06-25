import os
import re
import json
import pandas as pd
from flask import Flask, request, jsonify, Response
import urllib.request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "movies_master.csv")

print("Loading movies...")
try:
    df = pd.read_csv(DATA_PATH, nrows=20000)
    print(f"✓ {len(df)} movies loaded")
except Exception as e:
    print(f"ERROR: {e}")
    df = pd.DataFrame()

SESSIONS = {}

def is_hebrew(text):
    return bool(re.search(r"[\u0590-\u05FF]", str(text)))

def extract_genre(text):
    """Extract genre - flexible, not strict"""
    text = text.lower()
    genres = {
        "comedy": "Comedy", "קומדיה": "Comedy", "מצחיק": "Comedy", "funny": "Comedy",
        "drama": "Drama", "דרמה": "Drama", "מרגש": "Drama",
        "action": "Action", "אקשן": "Action", "fight": "Action",
        "horror": "Horror", "אימה": "Horror", "scary": "Horror",
        "romance": "Romance", "רומנטי": "Romance", "אהבה": "Romance",
        "thriller": "Thriller", "מתח": "Thriller", "suspense": "Thriller",
    }
    for kw, genre in genres.items():
        if kw in text:
            return genre
    return None

def extract_year(text):
    m = re.search(r"(19|20)\d{2}", text)
    if m:
        return int(m.group(0))
    return None

def get_recommendations(genre=None, year=None):
    try:
        filtered = df.copy()
        if genre:
            filtered = filtered[filtered["genres"].str.contains(genre, case=False, na=False)]
        if year:
            filtered = filtered[filtered["release_year"] >= year]
        
        if filtered.empty:
            return None
        
        filtered = filtered.sort_values("vote_average", ascending=False)
        row = filtered.iloc[0]
        
        return {
            "title": row.get("title", "Unknown"),
            "year": int(row.get("release_year", 0)) if row.get("release_year") else "?",
            "genres": row.get("genres", ""),
            "rating": round(float(row.get("vote_average", 0)), 1),
            "overview": (row.get("overview", "") or "")[:250]
        }
    except:
        return None

def call_openai(prompt):
    """Call OpenAI API"""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    
    try:
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.7
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
        
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"OpenAI error: {e}")
        return None

def should_start_interview(text):
    """Check if user wants to start movie recommendation"""
    keywords = ["סרט", "movie", "recommend", "המלץ", "start", "בואו", "let's", "find"]
    return any(kw in text.lower() for kw in keywords)

@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        user_msg = str(data.get("message", "")).strip()
        session_id = request.headers.get('X-Session-Id', 'default')
        
        if session_id not in SESSIONS:
            SESSIONS[session_id] = {
                "stage": 0,  # 0=casual, 1-4=interview questions, 5=done
                "lang": "he",
                "genre": None,
                "year": None,
                "occasion": None,
                "ref_movie": None,
            }
        
        session = SESSIONS[session_id]
        
        # Detect language from user message
        if user_msg:
            session["lang"] = "he" if is_hebrew(user_msg) else "en"
        
        lang = session["lang"]
        
        # Reset command
        if user_msg.lower() in ["סרט חדש", "new movie", "reset"]:
            SESSIONS[session_id] = {
                "stage": 0,
                "lang": lang,
                "genre": None,
                "year": None,
                "occasion": None,
                "ref_movie": None,
            }
            reply = "בואו נמצא לך סרט חדש! מה חדש?" if lang == "he" else "Let's find you a new movie! What's up?"
            return jsonify({"reply": reply, "stage": 0})
        
        if not user_msg:
            return jsonify({"reply": "", "stage": session["stage"]})
        
        # Stage 0: Casual chat until user wants to start
        if session["stage"] == 0:
            # Check if user wants to start interview
            if should_start_interview(user_msg):
                session["stage"] = 1
                reply = "יופי! אז, איזה סוג סרטים אתה אוהב?" if lang == "he" else "Cool! So, what kind of movies do you like?"
                return jsonify({"reply": reply, "stage": 1})
            
            # Otherwise, casual conversation with OpenAI
            prompt = f"User said: {user_msg}\nRespond casually and briefly in {'Hebrew' if lang == 'he' else 'English'}. Be friendly and conversational."
            reply = call_openai(prompt)
            if not reply:
                reply = "נשמע כיף! 😊" if lang == "he" else "Sounds good! 😊"
            
            return jsonify({"reply": reply, "stage": 0})
        
        # Stage 1: Genre question
        if session["stage"] == 1:
            genre = extract_genre(user_msg)
            if genre:
                session["genre"] = genre
                session["stage"] = 2
                reply = "מעניין! ומאיזה שנה?" if lang == "he" else "Nice! What year?"
                return jsonify({"reply": reply, "stage": 2})
            else:
                # Don't repeat - just accept it and move on
                session["genre"] = "Any"
                session["stage"] = 2
                reply = "אוקיי! ומאיזה שנה?" if lang == "he" else "Okay! What year?"
                return jsonify({"reply": reply, "stage": 2})
        
        # Stage 2: Year question
        if session["stage"] == 2:
            year = extract_year(user_msg)
            if year:
                session["year"] = year
            else:
                session["year"] = 2015  # Default
            
            session["stage"] = 3
            reply = "מה הרגשה? לדייט, להיות בודד, עם חברים?" if lang == "he" else "What's the vibe? Date, solo, with friends?"
            return jsonify({"reply": reply, "stage": 3})
        
        # Stage 3: Occasion question
        if session["stage"] == 3:
            session["occasion"] = user_msg  # Store whatever they said
            session["stage"] = 4
            reply = "יש סרט שאהבת שמתאים?" if lang == "he" else "Any movie you loved that fits?"
            return jsonify({"reply": reply, "stage": 4})
        
        # Stage 4: Reference movie or final recommendation
        if session["stage"] == 4:
            session["ref_movie"] = user_msg
            session["stage"] = 5
            
            # Get movie recommendation
            movie = get_recommendations(genre=session["genre"] if session["genre"] != "Any" else None, year=session["year"])
            
            if not movie:
                reply = "אה, לא מצאתי משהו מושלם. בואו נתחיל מחדש?" if lang == "he" else "Hmm, couldn't find a perfect match. Start over?"
                session["stage"] = 0
                return jsonify({"reply": reply, "stage": 0})
            
            # Use OpenAI to explain
            prompt = f"""Recommend this movie to user in 1-2 sentences:
Movie: {movie['title']} ({movie['year']})
Genre: {movie['genres']}
Rating: {movie['rating']}/10
Why: They like {session['genre']}, vibe: {session['occasion']}
Language: {'Hebrew' if lang == 'he' else 'English'}
Just recommend naturally, no extra text."""
            
            explanation = call_openai(prompt)
            if not explanation:
                explanation = f"הנה: {movie['title']} ({movie['year']}) ⭐{movie['rating']}" if lang == "he" else f"Here: {movie['title']} ({movie['year']}) ⭐{movie['rating']}"
            
            return jsonify({
                "reply": explanation,
                "movie": movie,
                "stage": 5
            })
        
        # Stage 5: After recommendation - casual chat
        if session["stage"] == 5:
            prompt = f"User asked: {user_msg}\nRespond briefly in {'Hebrew' if lang == 'he' else 'English'}. Keep it casual."
            reply = call_openai(prompt)
            if not reply:
                reply = "רוצה עוד המלצה? כתוב 'סרט חדש'" if lang == "he" else "Want another recommendation? Type 'new movie'"
            return jsonify({"reply": reply, "stage": 5})
        
        return jsonify({"reply": "שגיאה" if lang == "he" else "Error", "stage": session["stage"]})
    
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"reply": "שגיאה" if is_hebrew(str(data.get("message", ""))) else "Error"}), 500

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cinemate</title>
<link href="https://fonts.googleapis.com/css2?family=Heebo:wght@400;700;900&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #080808;
  --red: #d71920;
  --gold: #ffd166;
  --cream: #fff7ec;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  font-family: 'Heebo', sans-serif;
  direction: rtl;
  background: linear-gradient(rgba(0,0,0,.85), rgba(0,0,0,.85)), #080808;
  color: #f6f1ea;
  min-height: 100vh;
}

.marquee {
  position: fixed;
  top: 0;
  width: 100%;
  height: 8px;
  background: repeating-linear-gradient(90deg, var(--gold) 0 20px, #5b0004 20px 40px);
  z-index: 100;
}

header {
  text-align: center;
  padding: 25px 20px 15px;
  font-size: 32px;
  font-weight: 900;
  margin-top: 10px;
}

header span { color: var(--red); }

.stage-label {
  background: linear-gradient(90deg, #260003, #9e1018, #260003);
  color: var(--gold);
  padding: 12px;
  text-align: center;
  font-weight: 900;
  letter-spacing: 2px;
}

.container {
  max-width: 700px;
  margin: 20px auto;
  background: rgba(20, 20, 20, 0.9);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 0 20px 60px rgba(0,0,0,0.6);
}

.chat {
  height: 420px;
  overflow-y: auto;
  padding: 20px;
  background: var(--cream);
  color: #222;
}

.msg {
  margin: 12px 0;
  line-height: 1.6;
}

.msg.bot .bubble {
  background: #f0f0f0;
  padding: 12px 16px;
  border-radius: 14px;
  display: inline-block;
  max-width: 90%;
  word-wrap: break-word;
}

.msg.user {
  text-align: right;
}

.msg.user .bubble {
  background: linear-gradient(135deg, var(--red), #ff3040);
  color: white;
  padding: 12px 16px;
  border-radius: 14px;
  display: inline-block;
  max-width: 90%;
}

.movie-card {
  background: white;
  border: 2px solid var(--red);
  border-radius: 12px;
  padding: 14px;
  margin-top: 12px;
  margin-right: 0;
}

.movie-title {
  font-weight: 900;
  color: var(--red);
  font-size: 16px;
  margin-bottom: 6px;
}

.movie-meta {
  font-size: 13px;
  color: #666;
  margin-bottom: 8px;
}

.movie-overview {
  font-size: 13px;
  color: #444;
  line-height: 1.5;
}

.input-area {
  padding: 15px;
  display: flex;
  gap: 10px;
  background: rgba(20,20,20,0.95);
}

#inp {
  flex: 1;
  border: none;
  border-radius: 12px;
  padding: 12px;
  font-family: 'Heebo', sans-serif;
  font-size: 16px;
}

#btn {
  border: none;
  border-radius: 12px;
  padding: 12px 24px;
  background: linear-gradient(135deg, var(--red), #760006);
  color: white;
  font-weight: 900;
  cursor: pointer;
  font-family: 'Heebo', sans-serif;
}

#btn:hover { opacity: 0.9; }

::-webkit-scrollbar {
  width: 6px;
}

::-webkit-scrollbar-track {
  background: #f1f1f1;
}

::-webkit-scrollbar-thumb {
  background: #d71920;
  border-radius: 3px;
}
</style>
</head>
<body>

<div class="marquee"></div>

<header>
🎬 <span>Cinemate</span>
</header>

<div class="container">
  <div class="stage-label">FIND YOUR MOVIE</div>
  
  <div class="chat" id="chat">
    <div class="msg bot">
      <div class="bubble">שלום! 👋 מה חדש?</div>
    </div>
  </div>
  
  <div class="input-area">
    <input id="inp" placeholder="כתבו כאן..." autocomplete="off" spellcheck="false">
    <button id="btn">שלח</button>
  </div>
</div>

<script>
const chat = document.getElementById('chat');
const inp = document.getElementById('inp');
const btn = document.getElementById('btn');
const sessionId = 'sess_' + Math.random().toString(36).substr(2, 16);

function send() {
  const text = inp.value.trim();
  if (!text) return;
  
  const userDiv = document.createElement('div');
  userDiv.className = 'msg user';
  userDiv.innerHTML = '<div class="bubble">' + text.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</div>';
  chat.appendChild(userDiv);
  inp.value = '';
  chat.scrollTop = chat.scrollHeight;
  
  fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json', 'X-Session-Id': sessionId},
    body: JSON.stringify({message: text})
  })
  .then(r => r.json())
  .then(data => {
    const botDiv = document.createElement('div');
    botDiv.className = 'msg bot';
    
    let html = '<div class="bubble">' + (data.reply || '') + '</div>';
    
    if (data.movie) {
      html += `<div class="movie-card">
        <div class="movie-title">${data.movie.title}</div>
        <div class="movie-meta">${data.movie.year} • ⭐${data.movie.rating}/10 • ${data.movie.genres}</div>
        <div class="movie-overview">${data.movie.overview}</div>
      </div>`;
    }
    
    botDiv.innerHTML = html;
    chat.appendChild(botDiv);
    chat.scrollTop = chat.scrollHeight;
  })
  .catch(err => {
    const botDiv = document.createElement('div');
    botDiv.className = 'msg bot';
    botDiv.innerHTML = '<div class="bubble">שגיאה בתקשורת</div>';
    chat.appendChild(botDiv);
  });
}

btn.onclick = send;
inp.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });
</script>

</body>
</html>"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    print(f"\n{'='*60}")
    print(f"Cinemate on port {port}")
    print(f"Movies: {len(df)}")
    print(f"OpenAI: {'✓' if api_key else '✗'}")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
