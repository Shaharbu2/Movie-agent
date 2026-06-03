import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

# =========================
# 1. Load data
# =========================

df = pd.read_csv("movies_with_credits_clean.csv")

# =========================
# 2. Select features for anomaly detection
# =========================

# We look for movies that are unusual in their
# combination of: popularity, rating, votes, budget, revenue, runtime

features = [
    "popularity",
    "vote_average",
    "vote_count",
    "budget",
    "revenue",
    "runtime"
]

df_model = df[features].copy()
df_model = df_model.fillna(0)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_model)

# =========================
# 3. Isolation Forest
# =========================

iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,   # we expect ~5% of movies to be anomalies
    random_state=42
)

df["anomaly"] = iso.fit_predict(data_scaled)
df["anomaly_score"] = iso.decision_function(data_scaled)

# -1 = anomaly, 1 = normal
anomalies = df[df["anomaly"] == -1].copy()
normals   = df[df["anomaly"] ==  1].copy()

print(f"Total movies    : {len(df)}")
print(f"Normal movies   : {len(normals)}")
print(f"Anomalies found : {len(anomalies)}")

# =========================
# 4. Describe the anomalies
# =========================

print("\n" + "=" * 60)
print("ANOMALY TYPES – what makes them unusual?")
print("=" * 60)

# --- Type A: High popularity but low rating ---
type_a = anomalies[
    (anomalies["popularity"] > anomalies["popularity"].median()) &
    (anomalies["vote_average"] < 5.5)
].sort_values("popularity", ascending=False)

print(f"\nType A – Very popular but poorly rated ({len(type_a)} movies):")
print(type_a[["title","release_year","popularity","vote_average","genres_clean"]]
      .head(8).to_string(index=False))

# --- Type B: Extremely high budget but low revenue ---
type_b = anomalies[
    (anomalies["budget"] > 50_000_000) &
    (anomalies["revenue"] < anomalies["budget"] * 0.5)
].sort_values("budget", ascending=False)

print(f"\nType B – Big budget but box-office flops ({len(type_b)} movies):")
print(type_b[["title","release_year","budget","revenue","vote_average","genres_clean"]]
      .head(8).to_string(index=False))

# --- Type C: Very high rating but almost no votes (hidden gems) ---
type_c = anomalies[
    (anomalies["vote_average"] >= 7.5) &
    (anomalies["vote_count"] < 200)
].sort_values("vote_average", ascending=False)

print(f"\nType C – Hidden gems: high rating but few votes ({len(type_c)} movies):")
print(type_c[["title","release_year","vote_average","vote_count","genres_clean"]]
      .head(8).to_string(index=False))

# --- Type D: Unusually long or short runtime ---
type_d = anomalies[
    (anomalies["runtime"] > 180) | (anomalies["runtime"] < 60)
].sort_values("runtime")

print(f"\nType D – Unusual runtime (very long or very short) ({len(type_d)} movies):")
print(type_d[["title","release_year","runtime","genres_clean","vote_average"]]
      .head(8).to_string(index=False))

# =========================
# 5. Save results
# =========================

df[["id","title","release_year","genres_clean","popularity",
    "vote_average","vote_count","budget","revenue","runtime",
    "anomaly","anomaly_score"]].to_csv(
    "movies_with_anomalies.csv", index=False, encoding="utf-8-sig"
)

print("\nAnomaly detection completed. File saved: movies_with_anomalies.csv")
