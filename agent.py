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

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "movies_with_credits_clean.csv")

df = pd.read_csv(DATA_PATH)
df["overview"]       = df["overview"].fillna("")
df["genres_clean"]   = df["genres_clean"].fillna("")
df["keywords_clean"] = df["keywords_clean"].fillna("")
df["top_cast"]       = df["top_cast"].fillna("")
df["director"]       = df["director"].fillna("")

numeric_features = ["runtime","popularity","vote_average","vote_count","budget","revenue","genre_count"]
scaler = MinMaxScaler()
numeric_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_features]), columns=numeric_features)

def vectorize_column(col, max_features=None):
    vec = CountVectorizer(tokenizer=lambda x:[i.strip() for i in str(x).split(",")],
                          token_pattern=None, binary=True, max_features=max_features)
    m = vec.fit_transform(df[col].astype(str))
    return pd.DataFrame(m.toarray(), columns=[f"{col}_{c}" for c in vec.get_feature_names_out()])

genres_vec   = vectorize_column("genres_clean")
keywords_vec = vectorize_column("keywords_clean", max_features=100)
cluster_data = pd.concat([numeric_scaled, genres_vec, keywords_vec], axis=1)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(cluster_data)

CLUSTER_NAMES = {0:"Drama / Crime / History", 1:"Comedy / Romance",
                 2:"Action / Sci-Fi / Thriller", 3:"Family / Animation / Fantasy",
                 4:"Horror / Mystery / Thriller"}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

df["overview_clean"] = df["overview"].apply(clean_text)
tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(df["overview_clean"])

sim_data = pd.concat([numeric_scaled, genres_vec, vectorize_column("keywords_clean", max_features=300)], axis=1)
cosine_sim_matrix = cosine_similarity(sim_data)

iso_features = ["popularity","vote_average","vote_count","budget","revenue","runtime"]
iso_data = df[iso_features].fillna(0)
iso_scaler = MinMaxScaler()
iso_scaled = iso_scaler.fit_transform(iso_data)
iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
df["anomaly"]       = iso.fit_predict(iso_scaled)
df["anomaly_score"] = iso.decision_function(iso_scaled)

GENRE_KEYWORD_MAP = {
    "action":"Action","fight":"Action","battle":"Action","combat":"Action",
    "adventure":"Adventure","quest":"Adventure","journey":"Adventure",
    "animation":"Animation","animated":"Animation","cartoon":"Animation",
    "comedy":"Comedy","funny":"Comedy","humor":"Comedy","laugh":"Comedy","hilarious":"Comedy","fun":"Comedy",
    "crime":"Crime","criminal":"Crime","heist":"Crime","mafia":"Crime","gangster":"Crime","detective":"Crime",
    "documentary":"Documentary","true story":"Documentary",
    "drama":"Drama","emotional":"Drama","touching":"Drama",
    "family":"Family","kids":"Family","children":"Family","child":"Family",
    "fantasy":"Fantasy","magic":"Fantasy","magical":"Fantasy","wizard":"Fantasy","dragon":"Fantasy",
    "history":"History","historical":"History","ancient":"History","medieval":"History",
    "horror":"Horror","scary":"Horror","ghost":"Horror","haunted":"Horror","monster":"Horror","zombie":"Horror",
    "music":"Music","musical":"Music","singer":"Music",
    "mystery":"Mystery","mysterious":"Mystery","suspense":"Mystery","whodunit":"Mystery",
    "romance":"Romance","romantic":"Romance","love story":"Romance","love":"Romance",
    "science fiction":"Science Fiction","sci-fi":"Science Fiction","space":"Science Fiction",
    "robot":"Science Fiction","alien":"Science Fiction","future":"Science Fiction","time travel":"Science Fiction",
    "thriller":"Thriller","spy":"Thriller","assassin":"Thriller",
    "war":"War","soldier":"War","military":"War","army":"War","wwii":"War","world war":"War",
    "western":"Western","cowboy":"Western",
}

GENRE_TO_CLUSTER = {
    "Action":2,"Adventure":2,"Science Fiction":2,"Thriller":2,
    "Comedy":1,"Romance":1,"Music":1,
    "Drama":0,"Crime":0,"History":0,"War":0,"Documentary":0,"Western":0,
    "Family":3,"Animation":3,"Fantasy":3,
    "Horror":4,"Mystery":4,
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

def detect_intent(text):
    t = text.lower()
    for pat in [r"similar to (.+)", r"like (.+)", r"movies like (.+)", r"films like (.+)", r"something like (.+)"]:
        m = re.search(pat, t)
        if m:
            return "similar", m.group(1).strip().rstrip("?.,!")
    if any(k in t for k in ["unusual","anomaly","weird","flop","hidden gem","underrated","overrated","long movie","short movie"]):
        return "anomaly", text
    if any(k in t for k in ["what type","what kind","what genre","what cluster","category","group","classify"]):
        return "cluster_info", text
    return "search", text

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
    genre_scores   = df["genres_clean"].apply(genre_score).values
    cluster_scores = df["cluster"].apply(lambda c: 1.0 if c in matched_clusters else 0.0).values
    combined = 0.5*tfidf_scores + 0.3*genre_scores + 0.2*cluster_scores
    top_idx  = combined.argsort()[-top_n:][::-1]
    results = []
    for i, idx in enumerate(top_idx):
        row = df.iloc[idx]
        results.append({"rank":i+1,"title":row["title"],
            "year":int(row["release_year"]) if pd.notna(row["release_year"]) else "N/A",
            "genres":row["genres_clean"],"director":row["director"],
            "rating":round(float(row["vote_average"]),1),
            "overview":row["overview"][:200]+"..." if len(row["overview"])>200 else row["overview"],
            "score":round(float(combined[idx]),3)})
    reply = "I found " + str(top_n) + " movies matching your request"
    if matched_genres: reply += " (detected genres: " + ", ".join(matched_genres) + ")"
    reply += ":"
    return {"intent":"search","reply":reply,"results":results,"genres":matched_genres}

def handle_similar(movie_title, top_n=5):
    matches = df[df["title"].str.lower().str.contains(movie_title.lower(), na=False)]
    if matches.empty:
        return {"intent":"similar","reply":"Sorry, I couldn't find '"+movie_title+"' in my database.","results":[]}
    idx    = matches.index[0]
    found  = df.loc[idx,"title"]
    scores = cosine_sim_matrix[idx].copy()
    scores[idx] = -1
    top_idx = scores.argsort()[-top_n:][::-1]
    results = []
    for i, tidx in enumerate(top_idx):
        row = df.iloc[tidx]
        results.append({"rank":i+1,"title":row["title"],
            "year":int(row["release_year"]) if pd.notna(row["release_year"]) else "N/A",
            "genres":row["genres_clean"],"director":row["director"],
            "rating":round(float(row["vote_average"]),1),
            "overview":row["overview"][:200]+"..." if len(row["overview"])>200 else row["overview"],
            "score":round(float(scores[tidx]),3)})
    return {"intent":"similar","reply":"Here are "+str(top_n)+" movies similar to "+found+":","results":results}

def handle_anomaly(user_text, top_n=6):
    t = user_text.lower()
    anomalies = df[df["anomaly"]==-1].copy()
    if any(k in t for k in ["flop","budget","expensive","money"]):
        subset = anomalies[(anomalies["budget"]>50_000_000) & (anomalies["revenue"]<anomalies["budget"]*0.5)]
        label  = "big-budget box-office flops"
    elif any(k in t for k in ["hidden gem","underrated","unknown"]):
        subset = anomalies[(anomalies["vote_average"]>=7.5) &
                           (anomalies["vote_count"]<anomalies["vote_count"].quantile(0.4))]
        label  = "hidden gems (high rating, few votes)"
    elif any(k in t for k in ["long","short","runtime"]):
        subset = anomalies[(anomalies["runtime"]>180)|(anomalies["runtime"]<60)]
        label  = "movies with unusual runtime"
    elif any(k in t for k in ["popular","overrated"]):
        subset = anomalies[(anomalies["popularity"]>anomalies["popularity"].median()) &
                           (anomalies["vote_average"]<5.5)]
        label  = "popular but poorly rated movies"
    else:
        subset = anomalies.sort_values("anomaly_score").head(top_n)
        label  = "statistical anomalies in the dataset"
    subset = subset.head(top_n)
    results = []
    for i, (_, row) in enumerate(subset.iterrows()):
        results.append({"rank":i+1,"title":row["title"],
            "year":int(row["release_year"]) if pd.notna(row["release_year"]) else "N/A",
            "genres":row["genres_clean"],"director":row["director"],
            "rating":round(float(row["vote_average"]),1),
            "overview":row["overview"][:200]+"..." if len(row["overview"])>200 else row["overview"],
            "score":round(float(row["anomaly_score"]),3)})
    return {"intent":"anomaly","reply":"Here are "+label+":","results":results}

def handle_cluster_info(user_text):
    summary = []
    for c, name in CLUSTER_NAMES.items():
        subset = df[df["cluster"]==c]
        all_g  = ", ".join(subset["genres_clean"].dropna()).split(", ")
        top_g  = [g for g,_ in Counter(all_g).most_common(3) if g]
        summary.append({"cluster":c,"name":name,"count":len(subset),
            "top_genres":", ".join(top_g),
            "avg_rating":round(float(subset["vote_average"].mean()),2)})
    return {"intent":"cluster_info","reply":"Here is a breakdown of the 5 movie clusters:","clusters":summary}

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/api-key")
def api_key_route():
    return jsonify({"key": os.environ.get("ANTHROPIC_API_KEY","")})

def call_claude(user_text, results, intent):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or not results:
        return None
    try:
        import urllib.request
        movies = ""
        for r in results[:5]:
            movies += "- " + r["title"] + " (" + str(r["year"]) + "): " + r["genres"] + ", " + str(r["rating"]) + "/10\n"
        prompt = 'User asked: "' + user_text + '". Movies found:\n' + movies + 'Write 2-3 friendly sentences recommending these, mention 1-2 by name.'
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}]
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read())
            return data["content"][0]["text"]
    except Exception as e:
        print("Claude error:", e)
        return None

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_text = data.get("message","").strip()
    if not user_text:
        return jsonify({"reply":"Please type something!","results":[]})
    intent, payload = detect_intent(user_text)
    if intent == "similar":        result = handle_similar(payload)
    elif intent == "anomaly":      result = handle_anomaly(user_text)
    elif intent == "cluster_info": result = handle_cluster_info(user_text)
    else:                          result = handle_search(user_text)
    if intent != "cluster_info":
        claude_reply = call_claude(user_text, result.get("results", []), intent)
        if claude_reply:
            result["claude_reply"] = claude_reply
    return jsonify(result)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CineAgent</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0a0f;--surface:#13131a;--card:#1c1c26;--border:#2a2a38;--gold:#e8c96d;--gold2:#f5dfa0;--text:#e8e8f0;--muted:#7070a0;--accent:#5b8cff;--green:#6deba7}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;font-weight:300;height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{display:flex;align-items:center;gap:14px;padding:18px 28px;border-bottom:1px solid var(--border);background:var(--surface);flex-shrink:0}
.logo{font-family:'Bebas Neue',sans-serif;font-size:2rem;letter-spacing:3px;color:var(--gold);line-height:1}
.logo span{color:var(--text)}
.tagline{font-size:.75rem;color:var(--muted);letter-spacing:1px;text-transform:uppercase}
.pill{margin-left:auto;background:#1e2a1e;color:var(--green);font-size:.7rem;letter-spacing:1px;text-transform:uppercase;padding:4px 12px;border-radius:20px;border:1px solid #2a4a2a}
.main{display:flex;flex:1;overflow:hidden}
.sidebar{width:220px;flex-shrink:0;border-right:1px solid var(--border);background:var(--surface);padding:20px 16px;overflow-y:auto;display:flex;flex-direction:column;gap:8px}
.sidebar-title{font-size:.65rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:8px}
.suggestion{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:10px 12px;font-size:.78rem;color:var(--text);cursor:pointer;transition:all .2s;line-height:1.4}
.suggestion:hover{border-color:var(--gold);color:var(--gold)}
.s-tag{display:block;font-size:.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:3px}
.chat-wrap{flex:1;display:flex;flex-direction:column;overflow:hidden}
#messages{flex:1;overflow-y:auto;padding:24px 28px;display:flex;flex-direction:column;gap:20px;scrollbar-width:thin}
.msg{display:flex;flex-direction:column;gap:6px}
.msg.user{align-items:flex-end}
.msg.bot{align-items:flex-start}
.bubble{max-width:520px;padding:12px 16px;border-radius:14px;font-size:.88rem;line-height:1.6}
.msg.user .bubble{background:var(--accent);color:#fff;border-bottom-right-radius:4px}
.msg.bot .bubble{background:var(--card);border:1px solid var(--border);border-bottom-left-radius:4px}
.bubble b{color:var(--gold)}
.claude-bubble{max-width:520px;padding:12px 16px;border-radius:14px;font-size:.85rem;line-height:1.7;background:linear-gradient(135deg,#1a1630,#141020);border:1px solid rgba(232,201,109,.25);border-bottom-left-radius:4px;color:#c8c8e8;margin-top:4px}
.claude-label{font-size:.6rem;letter-spacing:2px;text-transform:uppercase;color:var(--gold);margin-bottom:6px}
.cards{display:flex;flex-direction:column;gap:10px;width:100%;max-width:680px;margin-top:6px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px 16px;display:flex;gap:14px;transition:border-color .2s}
.card:hover{border-color:var(--gold)}
.card-rank{font-family:'Bebas Neue',sans-serif;font-size:1.8rem;color:var(--border);line-height:1;min-width:32px;text-align:center}
.card:hover .card-rank{color:var(--gold)}
.card-body{flex:1}
.card-title{font-size:.95rem;font-weight:500;margin-bottom:3px}
.card-meta{font-size:.72rem;color:var(--muted);margin-bottom:6px;display:flex;gap:10px;flex-wrap:wrap}
.card-genres{font-size:.72rem;color:var(--gold);margin-bottom:6px}
.card-overview{font-size:.78rem;color:#9090b8;line-height:1.5}
.card-score{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;white-space:nowrap}
.rdot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}
.cluster-grid{display:flex;flex-wrap:wrap;gap:10px;max-width:680px;margin-top:6px}
.cluster-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px 16px;flex:1;min-width:160px}
.cluster-name{font-family:'Bebas Neue',sans-serif;font-size:1rem;letter-spacing:1px;color:var(--gold);margin-bottom:4px}
.cluster-stat{font-size:.72rem;color:var(--muted);line-height:1.7}
.typing{display:flex;gap:5px;padding:14px 16px;background:var(--card);border:1px solid var(--border);border-radius:14px;border-bottom-left-radius:4px;width:fit-content}
.dot{width:7px;height:7px;background:var(--muted);border-radius:50%;animation:bounce 1.2s infinite}
.dot:nth-child(2){animation-delay:.2s}.dot:nth-child(3){animation-delay:.4s}
@keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-6px);background:var(--gold)}}
.input-bar{padding:16px 28px;border-top:1px solid var(--border);background:var(--surface);display:flex;gap:10px;flex-shrink:0}
#inp{flex:1;background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px 16px;color:var(--text);font-family:'DM Sans',sans-serif;font-size:.88rem;outline:none;transition:border-color .2s}
#inp::placeholder{color:var(--muted)}
#inp:focus{border-color:var(--gold)}
#btn{background:var(--gold);color:#000;border:none;border-radius:12px;padding:0 22px;font-family:'Bebas Neue',sans-serif;font-size:1rem;letter-spacing:2px;cursor:pointer}
#btn:hover{background:var(--gold2)}
</style>
</head>
<body>
<header>
  <div>
    <div class="logo">Cine<span>Agent</span></div>
    <div class="tagline">AI-Powered Movie Intelligence</div>
  </div>
  <div class="pill">&#9679; Live</div>
</header>
<div class="main">
  <aside class="sidebar">
    <div class="sidebar-title">Try asking</div>
    <div class="suggestion" onclick="go('I want a funny romantic comedy')"><span class="s-tag">Search</span>Funny romantic comedy</div>
    <div class="suggestion" onclick="go('scary ghost movie with mystery')"><span class="s-tag">Search</span>Scary ghost mystery</div>
    <div class="suggestion" onclick="go('movies similar to Inception')"><span class="s-tag">Similar</span>Movies similar to Inception</div>
    <div class="suggestion" onclick="go('animated film for kids with magic')"><span class="s-tag">Search</span>Animated kids fantasy</div>
    <div class="suggestion" onclick="go('show me big budget box office flops')"><span class="s-tag">Anomaly</span>Big budget flops</div>
    <div class="suggestion" onclick="go('movies similar to The Dark Knight')"><span class="s-tag">Similar</span>Movies like The Dark Knight</div>
    <div class="suggestion" onclick="go('what are the movie clusters')"><span class="s-tag">Clusters</span>Movie clusters</div>
    <div class="suggestion" onclick="go('find me hidden gems with high rating')"><span class="s-tag">Anomaly</span>Hidden gems</div>
  </aside>
  <div class="chat-wrap">
    <div id="messages">
      <div class="msg bot">
        <div class="bubble">
          Welcome to <b>CineAgent</b> &#8212; your AI movie assistant.<br><br>
          I combine <b>Machine Learning</b> with <b>Claude AI</b> for smart recommendations.<br><br>
          &#8226; <b>Search</b> by mood or description<br>
          &#8226; <b>Find similar</b> movies<br>
          &#8226; <b>Detect anomalies</b> (flops, hidden gems)<br>
          &#8226; <b>Explore clusters</b><br><br>
          What are you in the mood for?
        </div>
      </div>
    </div>
    <div class="input-bar">
      <input id="inp" type="text" placeholder="Describe a movie, or ask anything..." autocomplete="off" />
      <button id="btn">SEND</button>
    </div>
  </div>
</div>
<script>
var M = document.getElementById('messages');
var I = document.getElementById('inp');
var B = document.getElementById('btn');

function rc(r) { return r>=7.5?'#6deba7':r>=6?'#e8c96d':'#ff6b6b'; }

function addMsg(role, html) {
  var d = document.createElement('div');
  d.className = 'msg ' + role;
  d.innerHTML = html;
  M.appendChild(d);
  M.scrollTop = M.scrollHeight;
}

function addTyping() {
  var d = document.createElement('div');
  d.className = 'msg bot';
  d.id = 'typ';
  d.innerHTML = '<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
  M.appendChild(d);
  M.scrollTop = M.scrollHeight;
}

function rmTyping() { var t=document.getElementById('typ'); if(t) t.remove(); }

function buildCards(results) {
  if (!results || !results.length) return '';
  var h = '<div class="cards">';
  for (var i=0; i<results.length; i++) {
    var r = results[i];
    h += '<div class="card">';
    h += '<div class="card-rank">' + r.rank + '</div>';
    h += '<div class="card-body">';
    h += '<div class="card-title">' + r.title + '</div>';
    h += '<div class="card-meta"><span>' + r.year + '</span><span>' + r.director + '</span>';
    h += '<span><span class="rdot" style="background:' + rc(r.rating) + '"></span>' + r.rating + '/10</span></div>';
    h += '<div class="card-genres">' + r.genres + '</div>';
    h += '<div class="card-overview">' + r.overview + '</div>';
    h += '</div><div class="card-score">score<br>' + r.score + '</div></div>';
  }
  h += '</div>';
  return h;
}

function buildClusters(clusters) {
  if (!clusters) return '';
  var h = '<div class="cluster-grid">';
  for (var i=0; i<clusters.length; i++) {
    var c = clusters[i];
    h += '<div class="cluster-card">';
    h += '<div class="cluster-name">' + c.name + '</div>';
    h += '<div class="cluster-stat">' + c.count + ' movies | Avg: ' + c.avg_rating + '<br>' + c.top_genres + '</div>';
    h += '</div>';
  }
  h += '</div>';
  return h;
}


function send() {
  var text = I.value.trim();
  if (!text) return;
  addMsg('user', '<div class="bubble">' + text + '</div>');
  I.value = '';
  addTyping();
  fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: text })
  })
  .then(function(r){ return r.json(); })
  .then(function(data){
    rmTyping();
    var extra = data.intent === 'cluster_info' ? buildClusters(data.clusters) : buildCards(data.results);
    addMsg('bot', '<div class="bubble">' + data.reply + '</div>' + extra);
    if (data.claude_reply) {
      setTimeout(function(){
        addMsg('bot', '<div class="claude-bubble"><div class="claude-label">&#10022; Claude AI</div>' + data.claude_reply + '</div>');
        M.scrollTop = M.scrollHeight;
      }, 300);
    }
  })
  .catch(function(e){
    rmTyping();
    addMsg('bot', '<div class="bubble">Something went wrong. Please try again.</div>');
  });
}

function go(q) { I.value = q; send(); }

B.onclick = function() { send(); };
I.onkeydown = function(e) { if (e.key === 'Enter') send(); };
</script>
</body>
</html>"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
