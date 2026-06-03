# CineAgent — AI Movie Assistant

An AI agent for movie recommendations built with Flask, deployed on Render.com.

## Features
- **Smart Search** — free text search combining TF-IDF + genre detection + cluster matching
- **Similar Movies** — find movies similar to any title using Cosine Similarity
- **Anomaly Detection** — discover flops, hidden gems, and unusual movies using Isolation Forest
- **Cluster Explorer** — browse the 5 auto-generated movie clusters using K-Means

## Dataset
TMDB 5000 Movies dataset (4,705 movies after cleaning)

## Project Structure
```
movie-agent/
├── data/
│   └── movies_with_credits_clean.csv
├── agent.py               # Main Flask app (run this)
├── clean_tmbd.py          # Data cleaning script
├── movie_similarity.py    # Similarity analysis
├── clustering.py          # K-Means clustering
├── nlp_overviews.py       # NLP on overviews
├── anomaly_detection.py   # Anomaly detection
├── smart_search.py        # Smart search module
├── requirements.txt
└── README.md
```

## Run Locally
```bash
pip install -r requirements.txt
python agent.py
# Open http://localhost:5000
```

## Deploy on Render.com
1. Push this folder to a GitHub repository
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Set:
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `gunicorn agent:app`
5. Click Deploy
