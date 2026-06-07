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

    # Generate Claude API natural language response
    matched_genres = result.get("genres", [])
    claude_reply = generate_claude_response(
        user_text, intent, result.get("results", []), matched_genres
    )
    if claude_reply:
        result["claude_reply"] = claude_reply

    return jsonify(result)

# ==============================================================
# 10. CLAUDE API — natural language response generator
# ==============================================================

import urllib.request

def generate_claude_response(user_text, intent, results, matched_genres=None):
    """
    Takes our search results and asks Claude API to write
    a natural, conversational recommendation response.
    """
    try:
        if intent == "cluster_info":
            return None  # clusters have their own UI, no need for Claude

        if not results:
            return None

        # Build a summary of results for Claude
        movies_summary = ""
        for r in results[:5]:
            movies_summary += f"- {r['title']} ({r['year']}) | {r['genres']} | Rating: {r['rating']}/10 | Director: {r['director']} | Overview: {r['overview'][:100]}\n"

        genre_hint = f"Detected genres: {', '.join(matched_genres)}" if matched_genres else ""

        if intent == "similar":
            prompt = f"""The user asked: "{user_text}"
We found these similar movies:
{movies_summary}
Write a short, enthusiastic 2-3 sentence response recommending these movies. 
Mention 1-2 specific titles by name and say WHY they match. Be conversational and friendly. No bullet points."""

        elif intent == "anomaly":
            prompt = f"""The user asked: "{user_text}"
We found these unusual/anomaly movies from our dataset:
{movies_summary}
Write a short, interesting 2-3 sentence response about these movies and what makes them unusual or special.
Mention 1-2 specific titles. Be conversational. No bullet points."""

        else:  # search
            prompt = f"""The user asked: "{user_text}"
{genre_hint}
We found these matching movies:
{movies_summary}
Write a short, enthusiastic 2-3 sentence response recommending these movies.
Mention 1-2 specific titles by name and say why they match the request. Be conversational and friendly. No bullet points."""

        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}]
        }

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
            return data["content"][0]["text"]

    except Exception as e:
        print(f"Claude API error: {e}")
        return None


# ==============================================================
# 11. HTML FRONTEND
# ==============================================================

HTML_PAGE = open(__file__.replace('agent.py','') + 'templates/index.html').read() if False else """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CineAgent — AI Movie Assistant</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&family=Playfair+Display:ital@0;1&display=swap" rel="stylesheet">
<style>
:root{--bg:#07070d;--surface:#0f0f18;--card:#16161f;--border:#252535;--gold:#e8c96d;--gold2:#f5dfa0;--text:#e8e8f0;--muted:#6060a0;--accent:#5b8cff;--green:#6deba7;--red:#ff6b6b}
*{margin:0;padding:0;box-sizing:border-box}html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;font-weight:300;overflow-x:hidden}
.hero{min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;position:relative;overflow:hidden;padding:40px 20px;text-align:center}
.hero-bg{position:absolute;inset:0;background:radial-gradient(ellipse 80% 60% at 50% 0%,#1a1030 0%,transparent 70%),radial-gradient(ellipse 60% 40% at 80% 80%,#0a1a2a 0%,transparent 60%),radial-gradient(ellipse 40% 60% at 20% 60%,#1a0a20 0%,transparent 60%);z-index:0}
.strip{position:absolute;width:3px;background:repeating-linear-gradient(to bottom,var(--gold) 0px,var(--gold) 20px,transparent 20px,transparent 30px);opacity:.15;animation:floatStrip 8s ease-in-out infinite}
.strip-1{left:8%;height:300px;top:10%}.strip-2{right:8%;height:200px;top:30%;animation-delay:2s}.strip-3{left:20%;height:150px;bottom:15%;animation-delay:4s}
@keyframes floatStrip{0%,100%{transform:translateY(0) rotate(2deg);opacity:.1}50%{transform:translateY(-20px) rotate(-1deg);opacity:.2}}
.hero-content{position:relative;z-index:1;max-width:800px}
.hero-eyebrow{display:inline-block;font-size:.7rem;letter-spacing:4px;text-transform:uppercase;color:var(--gold);border:1px solid rgba(232,201,109,.3);padding:6px 20px;border-radius:20px;margin-bottom:28px;animation:fadeUp .8s ease both}
.hero-title{font-family:'Bebas Neue',sans-serif;font-size:clamp(5rem,15vw,11rem);line-height:.9;letter-spacing:6px;color:var(--text);animation:fadeUp .8s ease .1s both}
.hero-title span{color:var(--gold);font-family:'Playfair Display',serif;font-style:italic;font-size:.55em;letter-spacing:2px;display:block;margin-top:8px}
.hero-sub{font-size:1rem;color:var(--muted);letter-spacing:1px;margin:24px 0 40px;line-height:1.7;animation:fadeUp .8s ease .2s both}
.hero-sub b{color:var(--text);font-weight:400}
.hero-cta{display:inline-flex;align-items:center;gap:10px;background:var(--gold);color:#000;font-family:'Bebas Neue',sans-serif;font-size:1.1rem;letter-spacing:3px;padding:16px 36px;border-radius:4px;text-decoration:none;transition:all .3s;animation:fadeUp .8s ease .3s both}
.hero-cta:hover{background:var(--gold2);transform:translateY(-2px);box-shadow:0 12px 40px rgba(232,201,109,.3)}
.hero-cta .arrow{font-size:1.3rem;transition:transform .3s}.hero-cta:hover .arrow{transform:translateX(4px)}
.features{padding:100px 20px;max-width:1100px;margin:0 auto}
.section-label{font-size:.65rem;letter-spacing:4px;text-transform:uppercase;color:var(--gold);margin-bottom:16px}
.section-title{font-family:'Bebas Neue',sans-serif;font-size:clamp(2.5rem,5vw,4rem);letter-spacing:3px;margin-bottom:60px;line-height:1}
.features-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:20px}
.feature-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:28px;transition:all .3s;position:relative;overflow:hidden}
.feature-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--gold),transparent);opacity:0;transition:opacity .3s}
.feature-card:hover{border-color:rgba(232,201,109,.3);transform:translateY(-4px)}.feature-card:hover::before{opacity:1}
.feature-icon{font-size:2rem;margin-bottom:16px}
.feature-name{font-family:'Bebas Neue',sans-serif;font-size:1.3rem;letter-spacing:2px;color:var(--gold);margin-bottom:8px}
.feature-desc{font-size:.82rem;color:var(--muted);line-height:1.7}
.examples{padding:60px 20px 100px;max-width:1100px;margin:0 auto}
.examples-grid{display:flex;flex-wrap:wrap;gap:10px}
.example-pill{background:var(--surface);border:1px solid var(--border);border-radius:30px;padding:10px 20px;font-size:.82rem;color:var(--text);cursor:pointer;transition:all .2s;text-decoration:none;display:inline-block}
.example-pill:hover{border-color:var(--gold);color:var(--gold);background:rgba(232,201,109,.05)}
.example-pill .tag{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-right:6px}
.chatbot-section{padding:60px 20px 100px;max-width:860px;margin:0 auto}
#chatbot{scroll-margin-top:40px}
.chat-container{background:var(--surface);border:1px solid var(--border);border-radius:24px;overflow:hidden;box-shadow:0 40px 120px rgba(0,0,0,.6)}
.chat-header{background:var(--card);border-bottom:1px solid var(--border);padding:18px 24px;display:flex;align-items:center;gap:12px}
.chat-header-logo{font-family:'Bebas Neue',sans-serif;font-size:1.3rem;letter-spacing:3px;color:var(--gold)}
.chat-header-logo span{color:var(--text)}
.chat-status{margin-left:auto;display:flex;align-items:center;gap:6px;font-size:.7rem;color:var(--green);letter-spacing:1px;text-transform:uppercase}
.status-dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(1.3)}}
#messages{height:480px;overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:18px;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.msg{display:flex;flex-direction:column;animation:fadeUp .3s ease}
.msg.user{align-items:flex-end}.msg.bot{align-items:flex-start}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.bubble{max-width:560px;padding:13px 17px;border-radius:16px;font-size:.88rem;line-height:1.65}
.msg.user .bubble{background:var(--accent);color:#fff;border-bottom-right-radius:4px}
.msg.bot .bubble{background:var(--card);border:1px solid var(--border);border-bottom-left-radius:4px}
.bubble b{color:var(--gold)}
.claude-bubble{max-width:560px;padding:13px 17px;border-radius:16px;font-size:.85rem;line-height:1.7;background:linear-gradient(135deg,#1a1630,#141020);border:1px solid rgba(232,201,109,.25);border-bottom-left-radius:4px;color:#c8c8e8;margin-top:6px}
.claude-label{font-size:.6rem;letter-spacing:2px;text-transform:uppercase;color:var(--gold);margin-bottom:6px;display:flex;align-items:center;gap:5px}
.claude-label::before{content:'✦';font-size:.7rem}
.cards{display:flex;flex-direction:column;gap:8px;width:100%;max-width:620px;margin-top:8px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:13px 15px;display:flex;gap:12px;transition:border-color .2s;animation:fadeUp .3s ease both}
.card:hover{border-color:rgba(232,201,109,.4)}
.card-rank{font-family:'Bebas Neue',sans-serif;font-size:1.6rem;color:var(--border);line-height:1;min-width:28px;text-align:center}
.card:hover .card-rank{color:var(--gold)}
.card-body{flex:1}
.card-title{font-size:.9rem;font-weight:500;margin-bottom:2px}
.card-meta{font-size:.7rem;color:var(--muted);display:flex;gap:8px;flex-wrap:wrap;margin-bottom:4px}
.card-genres{font-size:.7rem;color:var(--gold);margin-bottom:5px}
.card-overview{font-size:.76rem;color:#8080b0;line-height:1.5}
.card-score{font-size:.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;white-space:nowrap}
.rating-dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:3px}
.cluster-grid{display:flex;flex-wrap:wrap;gap:8px;max-width:620px;margin-top:8px}
.cluster-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px 16px;flex:1;min-width:150px;animation:fadeUp .3s ease both}
.cluster-name{font-family:'Bebas Neue',sans-serif;font-size:.95rem;letter-spacing:1px;color:var(--gold);margin-bottom:4px}
.cluster-stat{font-size:.7rem;color:var(--muted);line-height:1.7}
.typing{display:flex;gap:5px;padding:13px 16px;background:var(--card);border:1px solid var(--border);border-radius:16px;border-bottom-left-radius:4px;width:fit-content}
.dot{width:6px;height:6px;background:var(--muted);border-radius:50%;animation:bounce 1.2s infinite}
.dot:nth-child(2){animation-delay:.2s}.dot:nth-child(3){animation-delay:.4s}
@keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-6px);background:var(--gold)}}
.input-bar{padding:16px 20px;border-top:1px solid var(--border);background:var(--card);display:flex;gap:10px}
#input{flex:1;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 16px;color:var(--text);font-family:'DM Sans',sans-serif;font-size:.87rem;outline:none;transition:border-color .2s}
#input::placeholder{color:var(--muted)}#input:focus{border-color:var(--gold)}
#send{background:var(--gold);color:#000;border:none;border-radius:10px;padding:0 22px;font-family:'Bebas Neue',sans-serif;font-size:1rem;letter-spacing:2px;cursor:pointer;transition:all .2s}
#send:hover{background:var(--gold2)}#send:active{transform:scale(.97)}
footer{border-top:1px solid var(--border);padding:30px 20px;text-align:center;font-size:.75rem;color:var(--muted);letter-spacing:1px}
footer b{color:var(--gold)}
</style>
</head>
<body>
<section class="hero">
  <div class="hero-bg"></div>
  <div class="strip strip-1"></div><div class="strip strip-2"></div><div class="strip strip-3"></div>
  <div class="hero-content">
    <div class="hero-eyebrow">✦ AI-Powered Movie Intelligence ✦</div>
    <div class="hero-title">CineAgent<span>Your Personal Film Oracle</span></div>
    <p class="hero-sub">Powered by <b>Machine Learning</b> · <b>NLP</b> · <b>Claude AI</b><br>4,705 movies · Smart recommendations · Instant answers</p>
    <a href="#chatbot" class="hero-cta">Start Exploring <span class="arrow">→</span></a>
  </div>
</section>

<section class="features">
  <div class="section-label">What I Can Do</div>
  <div class="section-title">Four Ways to Discover Films</div>
  <div class="features-grid">
    <div class="feature-card"><div class="feature-icon">🔍</div><div class="feature-name">Smart Search</div><div class="feature-desc">Describe a mood, story, or feeling in plain English. TF-IDF + Genre Detection + Cluster Matching find the perfect fit.</div></div>
    <div class="feature-card"><div class="feature-icon">🎬</div><div class="feature-name">Similar Movies</div><div class="feature-desc">Type "movies similar to Inception" and get recommendations based on Cosine Similarity across genres, cast, and themes.</div></div>
    <div class="feature-card"><div class="feature-icon">🔴</div><div class="feature-name">Anomaly Detection</div><div class="feature-desc">Discover hidden gems, box-office flops, and statistically unusual films using Isolation Forest AI.</div></div>
    <div class="feature-card"><div class="feature-icon">📊</div><div class="feature-name">Cluster Explorer</div><div class="feature-desc">Browse 5 auto-generated movie clusters created by K-Means — from Action/Sci-Fi to Family/Animation.</div></div>
  </div>
</section>

<section class="examples">
  <div class="section-label">Try Asking</div>
  <div class="section-title">Example Questions</div>
  <div class="examples-grid">
    <a href="#chatbot" class="example-pill" onclick="setSuggestion('I want a scary horror movie with ghosts and mystery')"><span class="tag">Search</span>Scary ghost mystery</a>
    <a href="#chatbot" class="example-pill" onclick="setSuggestion('funny romantic comedy with love story')"><span class="tag">Search</span>Funny romantic comedy</a>
    <a href="#chatbot" class="example-pill" onclick="setSuggestion('movies similar to Inception')"><span class="tag">Similar</span>Movies like Inception</a>
    <a href="#chatbot" class="example-pill" onclick="setSuggestion('movies similar to The Dark Knight')"><span class="tag">Similar</span>Movies like The Dark Knight</a>
    <a href="#chatbot" class="example-pill" onclick="setSuggestion('show me big budget box office flops')"><span class="tag">Anomaly</span>Big budget flops</a>
    <a href="#chatbot" class="example-pill" onclick="setSuggestion('find me hidden gems with high rating')"><span class="tag">Anomaly</span>Hidden gems</a>
    <a href="#chatbot" class="example-pill" onclick="setSuggestion('animated film for kids with magic and dragons')"><span class="tag">Search</span>Animated kids fantasy</a>
    <a href="#chatbot" class="example-pill" onclick="setSuggestion('space adventure with aliens and robots')"><span class="tag">Search</span>Space sci-fi adventure</a>
    <a href="#chatbot" class="example-pill" onclick="setSuggestion('what are the movie clusters')"><span class="tag">Clusters</span>Explore clusters</a>
    <a href="#chatbot" class="example-pill" onclick="setSuggestion('soldier fighting in world war')"><span class="tag">Search</span>World war movie</a>
  </div>
</section>

<section class="chatbot-section">
  <div class="section-label">Talk to the Agent</div>
  <div class="section-title" style="margin-bottom:30px">Ask Anything</div>
  <div class="chat-container" id="chatbot">
    <div class="chat-header">
      <div class="chat-header-logo">Cine<span>Agent</span></div>
      <div class="chat-status"><div class="status-dot"></div>Online</div>
    </div>
    <div id="messages">
      <div class="msg bot">
        <div class="bubble">👋 Welcome to <b>CineAgent</b>!<br><br>I combine <b>Machine Learning</b> with <b>Claude AI</b> to give you smart, conversational movie recommendations.<br><br>Ask me anything — describe a mood, name a movie you love, or ask for something unusual. 🎬</div>
      </div>
    </div>
    <div class="input-bar">
      <input id="input" type="text" placeholder="Describe a movie, or ask anything..." autocomplete="off"/>
      <button id="send">Send</button>
    </div>
  </div>
</section>

<footer><b>CineAgent</b> · Built with Machine Learning, NLP &amp; Claude AI · 4,705 Movies · © 2025</footer>

<script>
const messagesEl=document.getElementById("messages"),inputEl=document.getElementById("input"),sendBtn=document.getElementById("send");
function ratingColor(r){return r>=7.5?"#6deba7":r>=6?"#e8c96d":"#ff6b6b"}
function renderCards(results){if(!results||!results.length)return"";return`<div class="cards">${results.map((r,i)=>`<div class="card" style="animation-delay:${i*.06}s"><div class="card-rank">${r.rank}</div><div class="card-body"><div class="card-title">${r.title}</div><div class="card-meta"><span>${r.year}</span><span>⬝ ${r.director}</span><span>⬝ <span class="rating-dot" style="background:${ratingColor(r.rating)}"></span>${r.rating}/10</span></div><div class="card-genres">${r.genres}</div><div class="card-overview">${r.overview}</div></div><div class="card-score">score<br>${r.score}</div></div>`).join("")}</div>`}
function renderClusters(clusters){if(!clusters)return"";return`<div class="cluster-grid">${clusters.map((c,i)=>`<div class="cluster-card" style="animation-delay:${i*.07}s"><div class="cluster-name">${c.name}</div><div class="cluster-stat">🎬 ${c.count} movies<br>⭐ Avg: ${c.avg_rating}<br>🎭 ${c.top_genres}</div></div>`).join("")}</div>`}
function addMessage(role,html,extra=""){const div=document.createElement("div");div.className=`msg ${role}`;div.innerHTML=`<div class="bubble">${html}</div>${extra}`;messagesEl.appendChild(div);messagesEl.scrollTop=messagesEl.scrollHeight}
function addClaudeReply(text){const div=document.createElement("div");div.className="msg bot";div.innerHTML=`<div class="claude-bubble"><div class="claude-label">Claude AI Analysis</div>${text}</div>`;messagesEl.appendChild(div);messagesEl.scrollTop=messagesEl.scrollHeight}
function addTyping(){const div=document.createElement("div");div.className="msg bot";div.id="typing";div.innerHTML=`<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>`;messagesEl.appendChild(div);messagesEl.scrollTop=messagesEl.scrollHeight}
function removeTyping(){const t=document.getElementById("typing");if(t)t.remove()}
async function sendMessage(text){
  if(!text.trim())return;
  addMessage("user",text);inputEl.value="";addTyping();
  try{
    const res=await fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:text})});
    const data=await res.json();removeTyping();
    const extra=data.intent==="cluster_info"?renderClusters(data.clusters):renderCards(data.results);
    addMessage("bot",data.reply,extra);
    if(data.claude_reply){setTimeout(()=>addClaudeReply(data.claude_reply),400)}
  }catch(e){removeTyping();addMessage("bot","⚠️ Something went wrong. Please try again.")}
}
function setSuggestion(q){inputEl.value=q;inputEl.focus()}
sendBtn.addEventListener("click",()=>sendMessage(inputEl.value));
inputEl.addEventListener("keydown",e=>{if(e.key==="Enter")sendMessage(inputEl.value)});
</script>
</body>
</html>"""

# ==============================================================
# 12. RUN
# ==============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
