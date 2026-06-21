# Copyright (c) 2026 Mohibul Hoque
# Licensed under the MIT License (see LICENSE file for details)
# Author: Mohibul Hoque <explorarhok@gmail.com> (github.com/speedyhok)

import pandas as pd

# Load our enriched data
df = pd.read_csv("data/processed/all_reviews_clean.csv")
df["posted_date"] = pd.to_datetime(df["posted_date"])
df["review_date"] = df["posted_date"].dt.date

# ── DAILY SENTIMENT TABLE ─────────────────────────────────────────────────
# For each game, for each day, calculate how positive/negative reviews were
# This is what we'll use to spot review bombs on a timeline

daily = df.groupby(["game_name", "review_date"]).agg(
    positive_count  = ("is_positive", "sum"),
    total_count     = ("is_positive", "count"),
    avg_suspicion   = ("suspicion_score", "mean"),
    avg_playtime    = ("playtime_at_review_hrs", "mean"),
).reset_index()

# Sentiment ratio: 1.0 = all positive, 0.0 = all negative
daily["sentiment_ratio"] = (daily["positive_count"] / daily["total_count"]).round(3)

# ── SPIKE DETECTOR ────────────────────────────────────────────────────────
# For each game, calculate a 7-day rolling average of negative reviews
# Flag any day where negatives jumped more than 3x that average

daily = daily.sort_values(["game_name", "review_date"])

daily["negative_count"] = daily["total_count"] - daily["positive_count"]
daily["rolling_avg_neg"] = (
    daily.groupby("game_name")["negative_count"]
    .transform(lambda x: x.rolling(7, min_periods=1).mean())
)
daily["is_spike"] = (
    (daily["negative_count"] > (daily["rolling_avg_neg"] * 3)) &
    (daily["sentiment_ratio"] < 0.5)
)

# ── SUMMARY ───────────────────────────────────────────────────────────────
print("── Review Bomb Spikes Detected ──")
spikes = daily[daily["is_spike"]].groupby("game_name")["is_spike"].count()
print(spikes.to_string())

print("\n── Overall Sentiment by Game ──")
overall = daily.groupby("game_name")["sentiment_ratio"].mean().round(3)
print(overall.to_string())

# ── SAVE ──────────────────────────────────────────────────────────────────
daily.to_csv("data/processed/daily_sentiment.csv", index=False)
print("\nSaved to data/processed/daily_sentiment.csv")