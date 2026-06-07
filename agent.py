import os
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from flask import Flask, request, jsonify, Response

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
        return {"intent":"similar","reply":"מצטער, לא מצאתי את הסרט '"+movie_title+"' במסד הנתונים שלי.","results":[]}
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
    return {"intent":"similar","reply":"הנה "+str(top_n)+" סרטים דומים ל-"+found+":","results":results}

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
    return {"intent":"anomaly","reply":"הנה "+label+":","results":results}

def handle_cluster_info(user_text):
    summary = []
    for c, name in CLUSTER_NAMES.items():
        subset = df[df["cluster"]==c]
        all_g  = ", ".join(subset["genres_clean"].dropna()).split(", ")
        top_g  = [g for g,_ in Counter(all_g).most_common(3) if g]
        summary.append({"cluster":c,"name":name,"count":len(subset),
            "top_genres":", ".join(top_g),
            "avg_rating":round(float(subset["vote_average"].mean()),2)})
    return {"intent":"cluster_info","reply":"הנה פירוט 5 קלסטרי הסרטים:","clusters":summary}

@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")

@app.route("/api-key")
def api_key_route():
    return jsonify({"key": os.environ.get("ANTHROPIC_API_KEY","")})

@app.route("/test-openai")
def test_openai():
    key = os.environ.get("OPENAI_API_KEY","")
    if not key:
        return jsonify({"status": "error", "message": "No OPENAI_API_KEY found in environment"})
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

def call_openai(user_text, results, intent):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or not results:
        return None
    try:
        import urllib.request
        movies = ""
        for r in results[:5]:
            movies += "- " + r["title"] + " (" + str(r["year"]) + "): " + r["genres"] + ", " + str(r["rating"]) + "/10\n"
        prompt = "ענה בעברית בלבד. המשתמש שאל: " + user_text + ". סרטים שנמצאו:\n" + movies + "כתוב 2-3 משפטים ידידותיים שממליצים על הסרטים, ציין 1-2 סרטים בשמם. טקסט פשוט בלבד ללא markdown."
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


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_text = data.get("message","").strip()
    if not user_text:
        return jsonify({"reply":"אנא הקלד משהו!","results":[]})
    intent, payload = detect_intent(user_text)
    if intent == "similar":        result = handle_similar(payload)
    elif intent == "anomaly":      result = handle_anomaly(user_text)
    elif intent == "cluster_info": result = handle_cluster_info(user_text)
    else:                          result = handle_search(user_text)
    if intent != "cluster_info":
        claude_reply = call_openai(user_text, result.get("results", []), intent)
        if claude_reply:
            result["claude_reply"] = claude_reply
    return jsonify(result)

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
  --accent:#3d6fff;--green:#3de0a0;--claude:#0e1128;
}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%}
body{
  background:var(--bg);color:var(--text);
  font-family:'Heebo',sans-serif;font-weight:300;
  direction:rtl;overflow:hidden;
  display:flex;flex-direction:column;
}

/* ---- BACKGROUND ---- */
.bg-wrap{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}
.bg-grad{
  position:absolute;inset:0;
  background:
    radial-gradient(ellipse 70% 50% at 50% -10%, rgba(61,111,255,0.18) 0%, transparent 60%),
    radial-gradient(ellipse 40% 40% at 85% 80%, rgba(232,184,75,0.07) 0%, transparent 55%),
    radial-gradient(ellipse 50% 60% at 15% 60%, rgba(61,111,255,0.06) 0%, transparent 55%);
}
.grid-lines{
  position:absolute;inset:0;
  background-image:
    linear-gradient(rgba(61,111,255,0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(61,111,255,0.04) 1px, transparent 1px);
  background-size:60px 60px;
}
.orb{
  position:absolute;border-radius:50%;filter:blur(80px);animation:float 8s ease-in-out infinite;
}
.orb1{width:400px;height:400px;background:rgba(61,111,255,0.08);top:-100px;left:50%;transform:translateX(-50%)}
.orb2{width:300px;height:300px;background:rgba(232,184,75,0.05);bottom:100px;right:10%;animation-delay:3s}
.orb3{width:250px;height:250px;background:rgba(61,200,255,0.04);top:40%;left:5%;animation-delay:5s}
@keyframes float{0%,100%{transform:translateY(0) translateX(-50%)}50%{transform:translateY(-30px) translateX(-50%)}}

/* ---- HEADER ---- */
header{
  position:relative;z-index:10;
  display:flex;align-items:center;justify-content:space-between;
  padding:16px 32px;
  border-bottom:1px solid rgba(61,111,255,0.15);
  background:rgba(6,9,20,0.8);
  backdrop-filter:blur(20px);
  flex-shrink:0;
}
.logo-wrap{display:flex;align-items:center;gap:14px}
.logo-icon{
  width:40px;height:40px;border-radius:12px;
  background:linear-gradient(135deg,#1a2a6c,#3d6fff);
  border:1px solid rgba(61,111,255,0.4);
  display:flex;align-items:center;justify-content:center;
  font-size:1.2rem;box-shadow:0 0 20px rgba(61,111,255,0.3);
}
.logo-text{font-size:1.3rem;font-weight:900;color:var(--text);letter-spacing:1px}
.logo-text span{color:var(--gold)}
.logo-sub{font-size:.65rem;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-top:1px}
.header-badge{
  display:flex;align-items:center;gap:6px;
  background:rgba(61,224,160,0.08);
  border:1px solid rgba(61,224,160,0.2);
  border-radius:20px;padding:5px 14px;
  font-size:.68rem;color:var(--green);font-weight:500;
}
.badge-dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(61,224,160,0.4)}50%{opacity:.7;box-shadow:0 0 0 4px rgba(61,224,160,0)}}

/* ---- MAIN LAYOUT ---- */
.main{position:relative;z-index:5;flex:1;display:flex;flex-direction:column;overflow:hidden}

/* ---- WELCOME SCREEN ---- */
#welcome{
  flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:40px 20px;text-align:center;gap:32px;
  animation:fadeIn .6s ease;
}
@keyframes fadeIn{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.welcome-icon{
  width:90px;height:90px;border-radius:24px;
  background:linear-gradient(135deg,#0d1a4a,#1a3a8f);
  border:1px solid rgba(61,111,255,0.35);
  display:flex;align-items:center;justify-content:center;
  font-size:2.5rem;
  box-shadow:0 0 60px rgba(61,111,255,0.25),inset 0 1px 0 rgba(255,255,255,0.05);
  animation:iconPulse 3s ease-in-out infinite;
}
@keyframes iconPulse{0%,100%{box-shadow:0 0 60px rgba(61,111,255,0.25)}50%{box-shadow:0 0 80px rgba(61,111,255,0.4)}}
.welcome-title{
  font-size:clamp(2rem,5vw,3.2rem);font-weight:900;
  background:linear-gradient(135deg,#dde2ff 0%,#e8b84b 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;line-height:1.1;
}
.welcome-sub{font-size:.95rem;color:var(--muted);line-height:1.8;max-width:520px}
.welcome-sub b{color:var(--text);font-weight:500}
.chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;max-width:600px}
.chip{
  background:rgba(255,255,255,0.04);border:1px solid var(--border);
  border-radius:24px;padding:8px 16px;font-size:.78rem;color:var(--muted);
  cursor:pointer;transition:all .2s;white-space:nowrap;
}
.chip:hover{border-color:var(--accent);color:var(--text);background:rgba(61,111,255,0.08)}
.chip .ctag{color:var(--gold);font-weight:700;margin-left:4px;font-size:.65rem}

/* ---- CHAT SCREEN ---- */
#chat-screen{
  flex:1;display:none;flex-direction:column;overflow:hidden;
}
#chat-screen.active{display:flex}
#messages{
  flex:1;overflow-y:auto;padding:24px 20%;
  display:flex;flex-direction:column;gap:20px;
  scrollbar-width:thin;scrollbar-color:var(--border) transparent;
}
@media(max-width:900px){#messages{padding:20px 5%}}

/* ---- MESSAGES ---- */
.msg{display:flex;flex-direction:column;gap:10px;animation:msgIn .3s ease}
@keyframes msgIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.msg.user{align-items:flex-start}
.msg.bot{align-items:flex-end}

.bubble{
  max-width:600px;padding:13px 18px;border-radius:18px;
  font-size:.9rem;line-height:1.7;
}
.msg.user .bubble{
  background:linear-gradient(135deg,#1a3080,#3d6fff);
  color:#fff;border-bottom-left-radius:4px;
  box-shadow:0 4px 20px rgba(61,111,255,0.3);
}
.msg.bot .bubble{
  background:var(--card);border:1px solid var(--border);
  border-bottom-right-radius:4px;
  box-shadow:0 2px 12px rgba(0,0,0,0.3);
}
.bubble b{color:var(--gold)}

/* CLAUDE BUBBLE */
.ai-box{
  max-width:600px;padding:16px 18px;border-radius:18px;border-bottom-right-radius:4px;
  background:var(--claude);
  border:1px solid rgba(232,184,75,0.2);
  font-size:.87rem;line-height:1.75;color:#b8c0e8;
  box-shadow:0 4px 24px rgba(0,0,0,0.4);
  position:relative;overflow:hidden;
}
.ai-box::before{
  content:'';position:absolute;top:0;right:0;left:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(232,184,75,0.4),transparent);
}
.ai-tag{
  font-size:.6rem;letter-spacing:2px;text-transform:uppercase;
  color:var(--gold);margin-bottom:10px;font-weight:700;
  display:flex;align-items:center;gap:6px;
}
.ai-tag::before{content:"✦";font-size:.75rem}

/* RESULT CARDS */
.cards{display:flex;flex-direction:column;gap:10px;width:100%;max-width:600px}
.card{
  background:rgba(17,19,32,0.8);
  border:1px solid var(--border);border-radius:14px;
  padding:14px 16px;display:flex;gap:14px;
  transition:all .2s;backdrop-filter:blur(10px);
}
.card:hover{
  border-color:rgba(61,111,255,0.4);
  transform:translateX(4px);
  box-shadow:0 4px 24px rgba(61,111,255,0.1);
}
.cnum{
  font-size:1.8rem;font-weight:900;color:var(--border);
  line-height:1;min-width:28px;text-align:center;transition:color .2s;
}
.card:hover .cnum{color:var(--gold)}
.cbody{flex:1;min-width:0}
.ctitle{font-size:.92rem;font-weight:700;margin-bottom:4px;color:var(--text)}
.cmeta{font-size:.68rem;color:var(--muted);display:flex;gap:8px;flex-wrap:wrap;margin-bottom:5px;align-items:center}
.cgenres{font-size:.68rem;color:var(--gold);margin-bottom:6px;font-weight:500}
.cdesc{font-size:.75rem;color:#5a6090;line-height:1.55}
.cscore{font-size:.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;white-space:nowrap;text-align:center}
.rdot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-left:3px}

/* CLUSTERS */
.clusters{display:flex;flex-wrap:wrap;gap:8px;max-width:600px}
.clu{
  background:var(--card);border:1px solid var(--border);
  border-radius:12px;padding:14px 16px;flex:1;min-width:150px;
}
.clu-name{font-size:.85rem;font-weight:700;color:var(--gold);margin-bottom:6px}
.clu-stat{font-size:.7rem;color:var(--muted);line-height:1.8}

/* TYPING */
.typing{
  display:flex;gap:5px;padding:14px 18px;
  background:var(--card);border:1px solid var(--border);
  border-radius:18px;border-bottom-right-radius:4px;width:fit-content;
}
.dot{width:6px;height:6px;background:var(--muted);border-radius:50%;animation:b 1.2s infinite}
.dot:nth-child(2){animation-delay:.2s}.dot:nth-child(3){animation-delay:.4s}
@keyframes b{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-6px);background:var(--gold)}}

/* ---- INPUT BAR ---- */
.input-wrap{
  position:relative;z-index:10;
  padding:20px 20%;
  background:rgba(6,9,20,0.7);backdrop-filter:blur(20px);
  border-top:1px solid rgba(61,111,255,0.1);
  flex-shrink:0;
}
@media(max-width:900px){.input-wrap{padding:16px 5%}}
.input-inner{
  display:flex;align-items:center;gap:0;
  background:rgba(17,19,32,0.9);
  border:1px solid rgba(61,111,255,0.25);
  border-radius:16px;padding:6px 6px 6px 16px;
  transition:border-color .2s;
  box-shadow:0 0 0 0 rgba(61,111,255,0);
}
.input-inner:focus-within{
  border-color:rgba(61,111,255,0.6);
  box-shadow:0 0 0 3px rgba(61,111,255,0.1);
}
#inp{
  flex:1;background:transparent;border:none;outline:none;
  color:var(--text);font-family:'Heebo',sans-serif;font-size:.92rem;
  direction:rtl;padding:8px 4px;
}
#inp::placeholder{color:var(--muted)}
#btn{
  background:linear-gradient(135deg,#2a50d0,#3d6fff);
  color:#fff;border:none;border-radius:11px;
  width:44px;height:44px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  font-size:1.1rem;transition:all .2s;flex-shrink:0;
  box-shadow:0 2px 12px rgba(61,111,255,0.4);
}
#btn:hover{background:linear-gradient(135deg,#3560e0,#5585ff);transform:scale(1.05)}
#btn:active{transform:scale(.95)}
.input-hint{text-align:center;margin-top:8px;font-size:.65rem;color:var(--muted)}
</style>
</head>
<body>
<div class="bg-wrap">
  <div class="bg-grad"></div>
  <div class="grid-lines"></div>
  <div class="orb orb1"></div>
  <div class="orb orb2"></div>
  <div class="orb orb3"></div>
</div>

<header>
  <div class="logo-wrap">
    <div class="logo-icon">&#127916;</div>
    <div>
      <div class="logo-text">Cine<span>Agent</span></div>
      <div class="logo-sub">AI Movie Intelligence</div>
    </div>
  </div>
  <div class="header-badge"><div class="badge-dot"></div>פעיל</div>
</header>

<div class="main">
  <!-- WELCOME SCREEN -->
  <div id="welcome">
    <div class="welcome-icon">&#127916;</div>
    <div>
      <div class="welcome-title">סינמה אייג&#x27;נט</div>
    </div>
    <div class="welcome-sub">
      עוזר חכם למציאת סרטים המשלב <b>למידת מכונה</b>, <b>עיבוד שפה טבעית</b> ו<b>Claude AI</b>.<br>
      מסד נתונים של <b>4,705 סרטים</b> &#x2014; חיפוש חופשי, המלצות, זיהוי חריגות וקלסטרים.
    </div>
    <div class="chips">
      <div class="chip" onclick="go('אני רוצה קומדיה רומנטית מצחיקה')"><span class="ctag">חיפוש</span>קומדיה רומנטית</div>
      <div class="chip" onclick="go('סרט אימה עם רוחות ומסתורין')"><span class="ctag">חיפוש</span>אימה ומסתורין</div>
      <div class="chip" onclick="go('movies similar to Inception')"><span class="ctag">דומה</span>דומה ל-Inception</div>
      <div class="chip" onclick="go('סרט אנימציה לילדים עם קסם')"><span class="ctag">חיפוש</span>אנימציה לילדים</div>
      <div class="chip" onclick="go('show me big budget box office flops')"><span class="ctag">חריגה</span>פלופים יקרים</div>
      <div class="chip" onclick="go('find me hidden gems with high rating')"><span class="ctag">חריגה</span>יהלומים נסתרים</div>
      <div class="chip" onclick="go('movies similar to The Dark Knight')"><span class="ctag">דומה</span>דומה ל-Dark Knight</div>
      <div class="chip" onclick="go('what are the movie clusters')"><span class="ctag">קלסטרים</span>הצג קלסטרים</div>
    </div>
  </div>

  <!-- CHAT SCREEN -->
  <div id="chat-screen">
    <div id="messages"></div>
  </div>

  <!-- INPUT -->
  <div class="input-wrap">
    <div class="input-inner">
      <input id="inp" type="text" placeholder="שאל על סרט, תאר מצב רוח, או בקש המלצה..." autocomplete="off" />
      <button id="btn">&#x2191;</button>
    </div>
    <div class="input-hint">Enter &#x21B5; לשליחה</div>
  </div>
</div>

<script>
var M   = document.getElementById('messages');
var I   = document.getElementById('inp');
var B   = document.getElementById('btn');
var WEL = document.getElementById('welcome');
var CS  = document.getElementById('chat-screen');
var started = false;

function showChat(){
  if(!started){
    started=true;
    WEL.style.display='none';
    CS.classList.add('active');
  }
}

function rc(r){return r>=7.5?'#3de0a0':r>=6?'#e8b84b':'#e85050';}

function addMsg(role,html){
  showChat();
  var d=document.createElement('div');
  d.className='msg '+role;
  d.innerHTML=html;
  M.appendChild(d);
  M.scrollTop=M.scrollHeight;
}

function addTyping(){
  showChat();
  var d=document.createElement('div');
  d.className='msg bot';d.id='typ';
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
    h+='<div class="cmeta"><span>'+r.year+'</span><span>'+r.director+'</span>';
    h+='<span><span class="rdot" style="background:'+rc(r.rating)+'"></span>'+r.rating+'/10</span></div>';
    h+='<div class="cgenres">'+r.genres+'</div>';
    h+='<div class="cdesc">'+r.overview+'</div>';
    h+='</div><div class="cscore">ציון<br>'+r.score+'</div></div>';
  }
  h+='</div>';
  return h;
}

function buildClusters(clusters){
  if(!clusters)return '';
  var h='<div class="clusters">';
  for(var i=0;i<clusters.length;i++){
    var c=clusters[i];
    h+='<div class="clu">';
    h+='<div class="clu-name">'+c.name+'</div>';
    h+='<div class="clu-stat">'+c.count+' סרטים | ממוצע: '+c.avg_rating+'<br>'+c.top_genres+'</div>';
    h+='</div>';
  }
  h+='</div>';
  return h;
}

function send(){
  var text=I.value.trim();
  if(!text)return;
  addMsg('user','<div class="bubble">'+text+'</div>');
  I.value='';
  addTyping();
  fetch('/chat',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({message:text})
  })
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
  .catch(function(e){
    rmTyping();
    addMsg('bot','<div class="bubble">משהו השתבש. נסה שוב.</div>');
  });
}

function go(q){I.value=q;send();}
B.onclick=function(){send();};
I.onkeydown=function(e){if(e.key==='Enter')send();};
</script>
</body>
</html>"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
