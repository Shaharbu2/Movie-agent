import os
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from flask import Flask, request, jsonify, render_template_string

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest

# ==============================================================
# APP SETUP
# ==============================================================

app = Flask(__name__)

# ==============================================================
# 1. LOAD & PREPARE DATA
# ==============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "movies_with_credits_clean.csv")

df = pd.read_csv(DATA_PATH)
df["overview"]       = df["overview"].fillna("")
df["genres_clean"]   = df["genres_clean"].fillna("")
df["keywords_clean"] = df["keywords_clean"].fillna("")
df["top_cast"]       = df["top_cast"].fillna("")
df["director"]       = df["director"].fillna("")

# ==============================================================
# 2. CLUSTERING
# ==============================================================

numeric_features = ["runtime","popularity","vote_average",
                    "vote_count","budget","revenue","genre_count"]

scaler = MinMaxScaler()
numeric_scaled = pd.DataFrame(
    scaler.fit_transform(df[numeric_features]),
    columns=numeric_features
)

def vectorize_column(col, max_features=None):
    vec = CountVectorizer(
        tokenizer=lambda x: [i.strip() for i in str(x).split(",")],
        token_pattern=None, binary=True, max_features=max_features
    )
    m = vec.fit_transform(df[col].astype(str))
    return pd.DataFrame(m.toarray(),
        columns=[f"{col}_{c}" for c in vec.get_feature_names_out()])

genres_vec   = vectorize_column("genres_clean")
keywords_vec = vectorize_column("keywords_clean", max_features=100)
cluster_data = pd.concat([numeric_scaled, genres_vec, keywords_vec], axis=1)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(cluster_data)

CLUSTER_NAMES = {
    0: "Drama / Crime / History",
    1: "Comedy / Romance",
    2: "Action / Sci-Fi / Thriller",
    3: "Family / Animation / Fantasy",
    4: "Horror / Mystery / Thriller"
}

# ==============================================================
# 3. TF-IDF ON OVERVIEWS
# ==============================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["overview_clean"] = df["overview"].apply(clean_text)

tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(df["overview_clean"])

# ==============================================================
# 4. COSINE SIMILARITY MATRIX (for similar-movie lookup)
# ==============================================================

cv_sim = CountVectorizer(
    tokenizer=lambda x: [i.strip() for i in str(x).split(",")],
    token_pattern=None, binary=True, max_features=300
)
cast_vec_sim     = cv_sim.fit_transform(df["top_cast"].astype(str))
director_vec_sim = cv_sim.fit_transform(df["director"].astype(str))
quality_vec_sim  = cv_sim.fit_transform(df["quality_category"].astype(str))

sim_data = pd.concat([
    numeric_scaled,
    genres_vec,
    vectorize_column("keywords_clean", max_features=300),
], axis=1)

cosine_sim_matrix = cosine_similarity(sim_data)

# ==============================================================
# 5. ANOMALY DETECTION
# ==============================================================

iso_features = ["popularity","vote_average","vote_count","budget","revenue","runtime"]
iso_data = df[iso_features].fillna(0)
iso_scaler = MinMaxScaler()
iso_scaled = iso_scaler.fit_transform(iso_data)

iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
df["anomaly"]       = iso.fit_predict(iso_scaled)
df["anomaly_score"] = iso.decision_function(iso_scaled)

# ==============================================================
# 6. GENRE KEYWORD MAP
# ==============================================================

GENRE_KEYWORD_MAP = {
    "action":"Action","fight":"Action","fighting":"Action","battle":"Action",
    "combat":"Action","explosion":"Action","chase":"Action",
    "adventure":"Adventure","quest":"Adventure","journey":"Adventure",
    "explore":"Adventure","expedition":"Adventure",
    "animation":"Animation","animated":"Animation","cartoon":"Animation",
    "comedy":"Comedy","funny":"Comedy","humor":"Comedy","laugh":"Comedy",
    "hilarious":"Comedy","fun":"Comedy",
    "crime":"Crime","criminal":"Crime","heist":"Crime","robbery":"Crime",
    "mafia":"Crime","gangster":"Crime","detective":"Crime",
    "documentary":"Documentary","real story":"Documentary","true story":"Documentary",
    "drama":"Drama","emotional":"Drama","serious":"Drama","touching":"Drama","moving":"Drama",
    "family":"Family","kids":"Family","children":"Family","child":"Family","wholesome":"Family",
    "fantasy":"Fantasy","magic":"Fantasy","magical":"Fantasy","wizard":"Fantasy",
    "dragon":"Fantasy","mythical":"Fantasy",
    "history":"History","historical":"History","ancient":"History","medieval":"History",
    "horror":"Horror","scary":"Horror","terrifying":"Horror","ghost":"Horror",
    "haunted":"Horror","monster":"Horror","zombie":"Horror",
    "music":"Music","musical":"Music","singer":"Music","band":"Music",
    "mystery":"Mystery","mysterious":"Mystery","suspense":"Mystery",
    "puzzle":"Mystery","whodunit":"Mystery",
    "romance":"Romance","romantic":"Romance","love story":"Romance",
    "love":"Romance","relationship":"Romance",
    "science fiction":"Science Fiction","sci-fi":"Science Fiction",
    "scifi":"Science Fiction","space":"Science Fiction","robot":"Science Fiction",
    "alien":"Science Fiction","future":"Science Fiction",
    "dystopia":"Science Fiction","time travel":"Science Fiction",
    "thriller":"Thriller","thrilling":"Thriller","tense":"Thriller",
    "suspenseful":"Thriller","spy":"Thriller","assassin":"Thriller",
    "war":"War","soldier":"War","military":"War","army":"War",
    "wwii":"War","world war":"War",
    "western":"Western","cowboy":"Western","wild west":"Western",
}

GENRE_TO_CLUSTER = {
    "Action":2,"Adventure":2,"Science Fiction":2,"Thriller":2,
    "Comedy":1,"Romance":1,"Music":1,
    "Drama":0,"Crime":0,"History":0,"War":0,"Documentary":0,"Western":0,
    "Family":3,"Animation":3,"Fantasy":3,
    "Horror":4,"Mystery":4,
}

def extract_genres(text):
    text_lower = text.lower()
    matched = set()
    for kw in sorted(GENRE_KEYWORD_MAP, key=len, reverse=True):
        if kw in text_lower:
            matched.add(GENRE_KEYWORD_MAP[kw])
    return list(matched)

def genres_to_clusters(genres):
    return {GENRE_TO_CLUSTER[g] for g in genres if g in GENRE_TO_CLUSTER}

# ==============================================================
# 7. AGENT INTENT DETECTION
# ==============================================================

def detect_intent(text):
    t = text.lower()

    # find similar to a specific movie
    similar_patterns = [
        r"similar to (.+)",
        r"like (.+)",
        r"movies like (.+)",
        r"films like (.+)",
        r"recommend.* like (.+)",
        r"something like (.+)",
    ]
    for pat in similar_patterns:
        m = re.search(pat, t)
        if m:
            return "similar", m.group(1).strip().rstrip("?.,!")

    # anomaly / unusual
    anomaly_keywords = ["unusual","anomaly","weird","strange","outlier",
                        "flop","hidden gem","underrated","overrated",
                        "big budget","long movie","short movie"]
    if any(k in t for k in anomaly_keywords):
        return "anomaly", text

    # cluster / what type
    cluster_keywords = ["what type","what kind","what genre","what cluster",
                        "category","group","classify"]
    if any(k in t for k in cluster_keywords):
        return "cluster_info", text

    # default: smart search
    return "search", text

# ==============================================================
# 8. HANDLER FUNCTIONS
# ==============================================================

def handle_search(user_text, top_n=5):
    matched_genres   = extract_genres(user_text)
    matched_clusters = genres_to_clusters(matched_genres)

    cleaned  = clean_text(user_text)
    user_vec = tfidf.transform([cleaned])
    tfidf_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()

    def genre_score(genre_str):
        if not matched_genres:
            return 0.0
        mg = [g.strip() for g in genre_str.split(",")]
        return sum(1 for g in matched_genres if g in mg) / len(matched_genres)

    genre_scores   = df["genres_clean"].apply(genre_score).values
    cluster_scores = df["cluster"].apply(
        lambda c: 1.0 if c in matched_clusters else 0.0
    ).values

    combined = 0.5*tfidf_scores + 0.3*genre_scores + 0.2*cluster_scores
    top_idx  = combined.argsort()[-top_n:][::-1]
    rows     = df.iloc[top_idx]

    results = []
    for i, (_, row) in enumerate(rows.iterrows()):
        results.append({
            "rank"         : i + 1,
            "title"        : row["title"],
            "year"         : int(row["release_year"]) if pd.notna(row["release_year"]) else "N/A",
            "genres"       : row["genres_clean"],
            "director"     : row["director"],
            "rating"       : round(float(row["vote_average"]), 1),
            "overview"     : row["overview"][:200] + "..." if len(row["overview"]) > 200 else row["overview"],
            "score"        : round(float(combined[top_idx[i]]), 3),
        })

    reply = f"I found {top_n} movies matching your request"
    if matched_genres:
        reply += f" (detected genres: {', '.join(matched_genres)})"
    reply += ":"

    return {"intent": "search", "reply": reply,
            "results": results, "genres": matched_genres}


def handle_similar(movie_title, top_n=5):
    matches = df[df["title"].str.lower().str.contains(movie_title.lower(), na=False)]
    if matches.empty:
        return {"intent":"similar","reply":f"Sorry, I couldn't find a movie called '{movie_title}' in my database.","results":[]}

    idx    = matches.index[0]
    found  = df.loc[idx, "title"]
    scores = cosine_sim_matrix[idx].copy()
    scores[idx] = -1
    top_idx = scores.argsort()[-top_n:][::-1]
    rows    = df.iloc[top_idx]

    results = []
    for i, (_, row) in enumerate(rows.iterrows()):
        results.append({
            "rank"    : i + 1,
            "title"   : row["title"],
            "year"    : int(row["release_year"]) if pd.notna(row["release_year"]) else "N/A",
            "genres"  : row["genres_clean"],
            "director": row["director"],
            "rating"  : round(float(row["vote_average"]), 1),
            "overview": row["overview"][:200] + "..." if len(row["overview"]) > 200 else row["overview"],
            "score"   : round(float(scores[top_idx[i]]), 3),
        })

    return {"intent":"similar",
            "reply": f"Here are {top_n} movies similar to **{found}**:",
            "results": results}


def handle_anomaly(user_text, top_n=6):
    t = user_text.lower()
    anomalies = df[df["anomaly"] == -1].copy()

    if any(k in t for k in ["flop","budget","expensive","money"]):
        subset = anomalies[(anomalies["budget"] > 50_000_000) &
                           (anomalies["revenue"] < anomalies["budget"] * 0.5)]
        label  = "big-budget box-office flops"
    elif any(k in t for k in ["hidden gem","underrated","unknown"]):
        subset = anomalies[(anomalies["vote_average"] >= 7.5) &
                           (anomalies["vote_count"] < anomalies["vote_count"].quantile(0.4))]
        label  = "hidden gems (high rating, few votes)"
    elif any(k in t for k in ["long","short","runtime"]):
        subset = anomalies[(anomalies["runtime"] > 180) | (anomalies["runtime"] < 60)]
        label  = "movies with unusual runtime"
    elif any(k in t for k in ["popular","overrated"]):
        subset = anomalies[(anomalies["popularity"] > anomalies["popularity"].median()) &
                           (anomalies["vote_average"] < 5.5)]
        label  = "popular but poorly rated movies"
    else:
        subset = anomalies.sort_values("anomaly_score").head(top_n)
        label  = "statistical anomalies in the dataset"

    subset = subset.head(top_n)
    results = []
    for i, (_, row) in enumerate(subset.iterrows()):
        results.append({
            "rank"    : i + 1,
            "title"   : row["title"],
            "year"    : int(row["release_year"]) if pd.notna(row["release_year"]) else "N/A",
            "genres"  : row["genres_clean"],
            "director": row["director"],
            "rating"  : round(float(row["vote_average"]), 1),
            "overview": row["overview"][:200] + "..." if len(row["overview"]) > 200 else row["overview"],
            "score"   : round(float(row["anomaly_score"]), 3),
        })

    return {"intent":"anomaly",
            "reply": f"Here are {label}:",
            "results": results}


def handle_cluster_info(user_text):
    summary = []
    for c, name in CLUSTER_NAMES.items():
        subset = df[df["cluster"] == c]
        all_g  = ", ".join(subset["genres_clean"].dropna()).split(", ")
        top_g  = [g for g, _ in Counter(all_g).most_common(3) if g]
        summary.append({
            "cluster": c,
            "name"   : name,
            "count"  : len(subset),
            "top_genres": ", ".join(top_g),
            "avg_rating": round(float(subset["vote_average"].mean()), 2),
        })
    return {"intent":"cluster_info",
            "reply": "Here's a breakdown of the 5 movie clusters in this dataset:",
            "clusters": summary}

# ==============================================================
# 9. FLASK ROUTES
# ==============================================================

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/api-key")
def api_key():
    return jsonify({"key": os.environ.get("ANTHROPIC_API_KEY", "")})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_text = data.get("message", "").strip()
    if not user_text:
        return jsonify({"reply": "Please type something!", "results": []})

    intent, payload = detect_intent(user_text)

    if intent == "similar":
        result = handle_similar(payload)
    elif intent == "anomaly":
        result = handle_anomaly(user_text)
    elif intent == "cluster_info":
        result = handle_cluster_info(user_text)
    else:
        result = handle_search(user_text)

    return jsonify(result)

# ==============================================================
# 11. HTML FRONTEND
# ==============================================================

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CineAgent</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0a0f;--surface:#13131a;--card:#1c1c26;--border:#2a2a38;--gold:#e8c96d;--gold2:#f5dfa0;--text:#e8e8f0;--muted:#7070a0;--accent:#5b8cff;--green:#6deba7}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:"DM Sans",sans-serif;font-weight:300;height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{display:flex;align-items:center;gap:14px;padding:18px 28px;border-bottom:1px solid var(--border);background:var(--surface);flex-shrink:0}
.logo{font-family:"Bebas Neue",sans-serif;font-size:2rem;letter-spacing:3px;color:var(--gold);line-height:1}
.logo span{color:var(--text)}
.tagline{font-size:.75rem;color:var(--muted);letter-spacing:1px;text-transform:uppercase}
.pill{margin-left:auto;background:#1e2a1e;color:var(--green);font-size:.7rem;letter-spacing:1px;text-transform:uppercase;padding:4px 12px;border-radius:20px;border:1px solid #2a4a2a}
.main{display:flex;flex:1;overflow:hidden}
.sidebar{width:220px;flex-shrink:0;border-right:1px solid var(--border);background:var(--surface);padding:20px 16px;overflow-y:auto;display:flex;flex-direction:column;gap:8px}
.sidebar-title{font-size:.65rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:8px;padding-left:4px}
.suggestion{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:10px 12px;font-size:.78rem;color:var(--text);cursor:pointer;transition:all .2s;line-height:1.4}
.suggestion:hover{border-color:var(--gold);color:var(--gold);background:#1e1c14}
.s-tag{display:block;font-size:.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:3px}
.chat-wrap{flex:1;display:flex;flex-direction:column;overflow:hidden}
#messages{flex:1;overflow-y:auto;padding:24px 28px;display:flex;flex-direction:column;gap:20px;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.msg{display:flex;flex-direction:column;gap:6px;animation:fadeUp .3s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.msg.user{align-items:flex-end}.msg.bot{align-items:flex-start}
.bubble{max-width:520px;padding:12px 16px;border-radius:14px;font-size:.88rem;line-height:1.6}
.msg.user .bubble{background:var(--accent);color:#fff;border-bottom-right-radius:4px}
.msg.bot .bubble{background:var(--card);border:1px solid var(--border);color:var(--text);border-bottom-left-radius:4px}
.bubble b{color:var(--gold)}
.claude-bubble{max-width:520px;padding:12px 16px;border-radius:14px;font-size:.85rem;line-height:1.7;background:linear-gradient(135deg,#1a1630,#141020);border:1px solid rgba(232,201,109,.25);border-bottom-left-radius:4px;color:#c8c8e8}
.claude-label{font-size:.6rem;letter-spacing:2px;text-transform:uppercase;color:var(--gold);margin-bottom:6px;display:flex;align-items:center;gap:5px}
.claude-label::before{content:"✦";font-size:.7rem}
.cards{display:flex;flex-direction:column;gap:10px;width:100%;max-width:680px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px 16px;display:flex;gap:14px;transition:border-color .2s;animation:fadeUp .3s ease both}
.card:hover{border-color:var(--gold)}
.card-rank{font-family:"Bebas Neue",sans-serif;font-size:1.8rem;color:var(--border);line-height:1;min-width:32px;text-align:center}
.card:hover .card-rank{color:var(--gold)}
.card-body{flex:1}
.card-title{font-size:.95rem;font-weight:500;margin-bottom:3px}
.card-meta{font-size:.72rem;color:var(--muted);margin-bottom:6px;display:flex;gap:10px;flex-wrap:wrap}
.card-genres{font-size:.72rem;color:var(--gold);margin-bottom:6px}
.card-overview{font-size:.78rem;color:#9090b8;line-height:1.5}
.card-score{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;white-space:nowrap}
.rating-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}
.cluster-grid{display:flex;flex-wrap:wrap;gap:10px;max-width:680px}
.cluster-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px 16px;flex:1;min-width:160px;animation:fadeUp .3s ease both}
.cluster-name{font-family:"Bebas Neue",sans-serif;font-size:1rem;letter-spacing:1px;color:var(--gold);margin-bottom:4px}
.cluster-stat{font-size:.72rem;color:var(--muted);line-height:1.7}
.typing{display:flex;gap:5px;padding:14px 16px;background:var(--card);border:1px solid var(--border);border-radius:14px;border-bottom-left-radius:4px;width:fit-content}
.dot{width:7px;height:7px;background:var(--muted);border-radius:50%;animation:bounce 1.2s infinite}
.dot:nth-child(2){animation-delay:.2s}.dot:nth-child(3){animation-delay:.4s}
@keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-6px);background:var(--gold)}}
.input-bar{padding:16px 28px;border-top:1px solid var(--border);background:var(--surface);display:flex;gap:10px;flex-shrink:0}
#input{flex:1;background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px 16px;color:var(--text);font-family:"DM Sans",sans-serif;font-size:.88rem;outline:none;transition:border-color .2s}
#input::placeholder{color:var(--muted)}
#input:focus{border-color:var(--gold)}
#send{background:var(--gold);color:#000;border:none;border-radius:12px;padding:0 22px;font-family:"Bebas Neue",sans-serif;font-size:1rem;letter-spacing:2px;cursor:pointer;transition:background .2s,transform .1s}
#send:hover{background:var(--gold2)}
#send:active{transform:scale(.97)}
</style>
</head>
<body>
<header>
  <div>
    <div class="logo">Cine<span>Agent</span></div>
    <div class="tagline">AI-Powered Movie Intelligence</div>
  </div>
  <div class="pill">&#x25CF; Live</div>
</header>
<div class="main">
  <aside class="sidebar">
    <div class="sidebar-title">Try asking</div>
    <div class="suggestion" onclick="sendSuggestion('I want a funny romantic comedy with a love story')"><span class="s-tag">Search</span>Funny romantic comedy</div>
    <div class="suggestion" onclick="sendSuggestion('scary ghost movie with mystery and suspense')"><span class="s-tag">Search</span>Scary ghost mystery</div>
    <div class="suggestion" onclick="sendSuggestion('movies similar to Inception')"><span class="s-tag">Similar</span>Movies similar to Inception</div>
    <div class="suggestion" onclick="sendSuggestion('animated film for kids with magic and fantasy')"><span class="s-tag">Search</span>Animated kids fantasy</div>
    <div class="suggestion" onclick="sendSuggestion('show me big budget box office flops')"><span class="s-tag">Anomaly</span>Big budget flops</div>
    <div class="suggestion" onclick="sendSuggestion('movies similar to The Dark Knight')"><span class="s-tag">Similar</span>Movies like The Dark Knight</div>
    <div class="suggestion" onclick="sendSuggestion('what are the movie clusters')"><span class="s-tag">Clusters</span>What are the clusters?</div>
    <div class="suggestion" onclick="sendSuggestion('find me hidden gems with high rating')"><span class="s-tag">Anomaly</span>Hidden gems</div>
  </aside>
  <div class="chat-wrap">
    <div id="messages">
      <div class="msg bot">
        <div class="bubble">
          &#x1F44B; Welcome to <b>CineAgent</b> &#x2014; your AI movie assistant.<br><br>
          I combine <b>Machine Learning</b> with <b>Claude AI</b> to give you smart recommendations.<br><br>
          &#x2022; <b>Search</b> by mood or description<br>
          &#x2022; <b>Find similar</b> movies ("movies similar to Inception")<br>
          &#x2022; <b>Detect anomalies</b> (flops, hidden gems)<br>
          &#x2022; <b>Explore clusters</b> &#x2014; how movies are grouped<br><br>
          What are you in the mood for?
        </div>
      </div>
    </div>
    <div class="input-bar">
      <input id="input" type="text" placeholder="Describe a movie, or ask anything..." autocomplete="off" />
      <button id="send">Send</button>
    </div>
  </div>
</div>
<script>
var msgs = document.getElementById("messages");
var inp  = document.getElementById("input");
var btn  = document.getElementById("send");

function ratingColor(r) {
  if (r >= 7.5) return "#6deba7";
  if (r >= 6)   return "#e8c96d";
  return "#ff6b6b";
}

function renderCards(results) {
  if (!results || !results.length) return "";
  var html = '<div class="cards">';
  for (var i = 0; i < results.length; i++) {
    var r = results[i];
    html += '<div class="card" style="animation-delay:' + (i * 0.06) + 's">';
    html += '<div class="card-rank">' + r.rank + '</div>';
    html += '<div class="card-body">';
    html += '<div class="card-title">' + r.title + '</div>';
    html += '<div class="card-meta"><span>' + r.year + '</span><span>&sdot; ' + r.director + '</span>';
    html += '<span>&sdot; <span class="rating-dot" style="background:' + ratingColor(r.rating) + '"></span>' + r.rating + '/10</span></div>';
    html += '<div class="card-genres">' + r.genres + '</div>';
    html += '<div class="card-overview">' + r.overview + '</div>';
    html += '</div><div class="card-score">score<br>' + r.score + '</div></div>';
  }
  html += '</div>';
  return html;
}

function renderClusters(clusters) {
  if (!clusters) return "";
  var html = '<div class="cluster-grid">';
  for (var i = 0; i < clusters.length; i++) {
    var c = clusters[i];
    html += '<div class="cluster-card" style="animation-delay:' + (i * 0.07) + 's">';
    html += '<div class="cluster-name">' + c.name + '</div>';
    html += '<div class="cluster-stat">&#x1F3AC; ' + c.count + ' movies<br>&#x2B50; Avg: ' + c.avg_rating + '<br>&#x1F3AD; ' + c.top_genres + '</div>';
    html += '</div>';
  }
  html += '</div>';
  return html;
}

function addMessage(role, html, extra) {
  var div = document.createElement("div");
  div.className = "msg " + role;
  div.innerHTML = '<div class="bubble">' + html + '</div>' + (extra || "");
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function addClaudeReply(text) {
  var div = document.createElement("div");
  div.className = "msg bot";
  div.innerHTML = '<div class="claude-bubble"><div class="claude-label">Claude AI Analysis</div>' + text + '</div>';
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function addTyping() {
  var div = document.createElement("div");
  div.className = "msg bot";
  div.id = "typing";
  div.innerHTML = '<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function removeTyping() {
  var t = document.getElementById("typing");
  if (t) t.remove();
}

function callClaudeAPI(userText, results, intent) {
  if (!results || results.length === 0) return;
  fetch("/api-key").then(function(r) { return r.json(); }).then(function(kd) {
    var apiKey = kd.key || "";
    if (!apiKey) return;
    var movieList = "";
    for (var i = 0; i < Math.min(results.length, 5); i++) {
      movieList += "- " + results[i].title + " (" + results[i].year + "): " + results[i].genres + ", rated " + results[i].rating + "/10
";
    }
    var prompt = intent === "similar"
      ? "The user asked: " + userText + ". We found these similar movies:
" + movieList + "Write 2-3 enthusiastic sentences recommending these. Mention 1-2 by name. Be conversational."
      : intent === "anomaly"
      ? "The user asked: " + userText + ". We found these unusual movies:
" + movieList + "Write 2-3 interesting sentences about what makes them unusual. Mention 1-2 by name."
      : "The user asked: " + userText + ". We found these matching movies:
" + movieList + "Write 2-3 enthusiastic sentences recommending these. Mention 1-2 by name. Be conversational.";
    fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01"
      },
      body: JSON.stringify({
        model: "claude-sonnet-4-20250514",
        max_tokens: 200,
        messages: [{ role: "user", content: prompt }]
      })
    }).then(function(r) { return r.json(); }).then(function(data) {
      if (data.content && data.content[0]) addClaudeReply(data.content[0].text);
    }).catch(function(e) { console.log("Claude API error:", e); });
  }).catch(function(e) { console.log("Key fetch error:", e); });
}

function sendMessage(text) {
  if (!text || !text.trim()) return;
  addMessage("user", text);
  inp.value = "";
  addTyping();
  fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text })
  }).then(function(r) { return r.json(); }).then(function(data) {
    removeTyping();
    var extra = data.intent === "cluster_info" ? renderClusters(data.clusters) : renderCards(data.results);
    addMessage("bot", data.reply, extra);
    if (data.intent !== "cluster_info") {
      setTimeout(function() { callClaudeAPI(text, data.results, data.intent); }, 400);
    }
  }).catch(function(e) {
    removeTyping();
    addMessage("bot", "Something went wrong. Please try again.");
  });
}

function sendSuggestion(q) { sendMessage(q); }

btn.addEventListener("click", function() { sendMessage(inp.value); });
inp.addEventListener("keydown", function(e) { if (e.key === "Enter") sendMessage(inp.value); });
</script>
</body>
</html>
"""




# ==============================================================
# 12. RUN
# ==============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
