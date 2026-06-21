# Copyright (c) 2026 Mohibul Hoque
# Licensed under the MIT License (see LICENSE file for details)
# Author: Mohibul Hoque <explorarhok@gmail.com> (github.com/speedyhok)

import pandas as pd

# Load the combined reviews file we collected in Script 1
df = pd.read_csv("data/raw/all_reviews.csv")
print(f"Loaded {len(df)} reviews")

# ── FEATURE 1: What hour of the day was the review posted ──────────────────
# Bots tend to post in bursts at odd hours
df["posted_date"] = pd.to_datetime(df["posted_date"])
df["review_hour"] = df["posted_date"].dt.hour
df["review_day_of_week"] = df["posted_date"].dt.day_name()

# ── FEATURE 2: How many days after launch was this posted ──────────────────
# Day 0, Day 1 etc. — launch window reviews are the most manipulated
launch_dates = {
    "Cyberpunk 2077":        "2020-12-10",
    "No Man's Sky":          "2016-08-12",
    "Elden Ring":            "2022-02-25",
    "Battlefield 2042":      "2021-11-19",
    "Hogwarts Legacy":       "2023-02-10",
    "Starfield":             "2023-09-06",
    "Fallout 76":            "2018-11-14",
    "Hades":                 "2020-09-17",
    "Baldurs Gate 3":        "2023-08-03",
    "Stardew Valley":        "2016-02-26",
    "The Last of Us Part I": "2023-03-28",
    "Disco Elysium":         "2019-10-15",
    "Dying Light 2":         "2022-02-04",
    "Gotham Knights":        "2022-10-21",
    "Forspoken":             "2023-01-24",
    "Red Dead Redemption 2": "2019-12-05",
    "Counter-Strike 2":      "2023-09-27",
    "GTA V":                 "2015-04-14",
    "Hollow Knight":         "2017-02-24",
    "Dead by Daylight":      "2016-06-14",

}

df["launch_date"] = df["game_name"].map(launch_dates)
df["launch_date"] = pd.to_datetime(df["launch_date"])
df["days_since_launch"] = (df["posted_date"] - df["launch_date"]).dt.days
df["is_launch_window"] = df["days_since_launch"] <= 7

# ── FEATURE 3: Playtime flags ──────────────────────────────────────────────
# Reviewing a game after less than 1 hour is a strong manipulation signal
df["is_low_playtime"] = df["playtime_at_review_hrs"] < 1

# Ratio of playtime when reviewed vs total playtime ever
# A ratio close to 1.0 means they reviewed immediately and never played again
df["playtime_ratio"] = df["playtime_at_review_hrs"] / (df["playtime_forever_hrs"] + 0.01)

# ── FEATURE 4: Account flags ───────────────────────────────────────────────
# First review ever = throwaway account
df["is_first_review"] = df["num_reviews_by_author"] == 1

# Thin account = owns fewer than 5 games total
df["is_thin_account"] = df["num_games_owned"] < 5

# ── FEATURE 5: Helpfulness ─────────────────────────────────────────────────
# Reviews the community found helpful are more likely genuine
df["helpfulness_score"] = pd.to_numeric(df["helpfulness_score"], errors="coerce").fillna(0)

# ── SUSPICION SCORE ────────────────────────────────────────────────────────
# Weighted formula combining all signals into one number 0-100
# Higher = more suspicious
df["suspicion_score"] = (
    (df["is_low_playtime"].astype(int)   * 25) +
    (df["is_first_review"].astype(int)   * 20) +
    (df["is_thin_account"].astype(int)   * 20) +
    (df["received_for_free"].astype(int) * 15) +
    ((1 - df["helpfulness_score"])       * 20)
)

# ── LABEL FOR MACHINE LEARNING ─────────────────────────────────────────────
# We'll use this later when we train the Random Forest model
df["is_suspicious"] = (df["suspicion_score"] >= 60).astype(int)

# ── SUMMARY ───────────────────────────────────────────────────────────────
print("\n── Suspicion Score Summary by Game ──")
summary = df.groupby("game_name")["suspicion_score"].agg(["mean", "max", "count"])
summary.columns = ["avg_suspicion", "max_suspicion", "total_reviews"]
summary["avg_suspicion"] = summary["avg_suspicion"].round(1)
print(summary)

print("\n── Suspicious Review % by Game ──")
pct = df.groupby("game_name")["is_suspicious"].mean() * 100
print(pct.round(1).to_string())

print("\n── Low Playtime Reviews % by Game ──")
low = df.groupby("game_name")["is_low_playtime"].mean() * 100
print(low.round(1).to_string())

# ── SAVE ──────────────────────────────────────────────────────────────────
df.to_csv("data/processed/all_reviews_clean.csv", index=False)
print(f"\nSaved enriched data to data/processed/all_reviews_clean.csv")