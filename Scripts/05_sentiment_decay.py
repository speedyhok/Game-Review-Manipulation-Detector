# Copyright (c) 2026 Mohibul Hoque
# Licensed under the MIT License (see LICENSE file for details)
# Author: Mohibul Hoque <explorarhok@gmail.com> (github.com/speedyhok)

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ── LOAD KAGGLE DATA ──────────────────────────────────────────────────────
print("Loading Kaggle data...")
games_df = pd.read_csv("data/raw/games.csv")
reviews_df = pd.read_csv("data/raw/games_reviews.csv", low_memory=False)
print(f"Games: {len(games_df)} | Reviews: {len(reviews_df)}")

# ── OUR TARGET GAMES ─────────────────────────────────────────────────────
TARGET_GAMES = {
    "Cyberpunk 2077":        1300144696,
    "No Man's Sky":          1300223342,
    "Elden Ring":            1300501979,
    "Battlefield 2042":      1300538028,
    "Hogwarts Legacy":       1300523817,
    "Starfield":             1300486989,
    "Fallout 76":            1300486337,
    "Hades":                 1300494289,
    "Baldurs Gate 3":        1300501848,
    "Stardew Valley":        1300170113,
    "The Last of Us Part I": 1300572186,
    "Disco Elysium":         1300468042,
    "Dying Light 2":         1300486966,
    "Gotham Knights":        1300522308,
    "Forspoken":             1300518494,
    "Red Dead Redemption 2": 1300461445,
    "Counter-Strike 2":      1300602226,
    "Hollow Knight":         1300450663,
    "Dead by Daylight":      1300454843,
}

LAUNCH_DATES = {
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
    "The Last of Us Part I": "2022-09-02",
    "Disco Elysium":         "2019-10-15",
    "Dying Light 2":         "2022-02-04",
    "Gotham Knights":        "2022-10-21",
    "Forspoken":             "2023-01-24",
    "Red Dead Redemption 2": "2018-10-26",
    "Counter-Strike 2":      "2023-09-27",
    "Hollow Knight":         "2017-02-24",
    "Dead by Daylight":      "2016-06-14",
}

# ── FILTER TO OUR GAMES ───────────────────────────────────────────────────
target_ids = list(TARGET_GAMES.values())
game_reviews = reviews_df[reviews_df["id"].isin(target_ids)].copy()

# Add friendly game name
id_to_name = {v: k for k, v in TARGET_GAMES.items()}
game_reviews["game_name"] = game_reviews["id"].map(id_to_name)

# User reviews only, with valid scores
game_reviews = game_reviews[game_reviews["review_type"] == "user"]
game_reviews = game_reviews[game_reviews["score"].notna()]
game_reviews["date"] = pd.to_datetime(game_reviews["date"], errors="coerce")
game_reviews = game_reviews[game_reviews["date"].notna()]

print(f"\nFiltered to {len(game_reviews)} user reviews across our target games")
print("\nReviews per game:")
print(game_reviews["game_name"].value_counts().to_string())

# ── NORMALIZE SCORES TO 0-1 ───────────────────────────────────────────────
# Kaggle scores are 0-100, convert to sentiment ratio like our Steam data
game_reviews["sentiment_ratio"] = game_reviews["score"] / 100

# ── ADD MONTHS SINCE LAUNCH ───────────────────────────────────────────────
game_reviews["launch_date"] = game_reviews["game_name"].map(LAUNCH_DATES)
game_reviews["launch_date"] = pd.to_datetime(game_reviews["launch_date"])
game_reviews["days_since_launch"] = (
    game_reviews["date"] - game_reviews["launch_date"]
).dt.days
game_reviews["months_since_launch"] = (game_reviews["days_since_launch"] / 30).astype(int)

# Only keep reviews after launch
game_reviews = game_reviews[game_reviews["days_since_launch"] >= 0]

# ── MONTHLY AGGREGATION ───────────────────────────────────────────────────
monthly = game_reviews.groupby(["game_name", "months_since_launch"]).agg(
    avg_score      = ("sentiment_ratio", "mean"),
    review_count   = ("score", "count"),
    median_score   = ("sentiment_ratio", "median"),
).reset_index()

monthly = monthly[monthly["review_count"] >= 3]

print("\nMonths of data per game:")
print(monthly.groupby("game_name")["months_since_launch"].max().sort_values().to_string())

# ── TRAJECTORY CLASSIFIER ─────────────────────────────────────────────────
results = []

for game in monthly["game_name"].unique():
    gdf = monthly[monthly["game_name"] == game].sort_values("months_since_launch")

    if len(gdf) < 4:
        print(f"  Skipping {game} — not enough monthly data ({len(gdf)} months)")
        continue

    sentiment = gdf["avg_score"].values
    months    = gdf["months_since_launch"].values

    # Linear trend
    X = months.reshape(-1, 1)
    reg = LinearRegression().fit(X, sentiment)
    slope = reg.coef_[0]

    # Early vs late sentiment
    early = sentiment[:3].mean()
    late  = sentiment[-3:].mean()
    overall_avg = sentiment.mean()
    variance    = sentiment.std()

    # Launch month score (month 0 or 1)
    launch_data = gdf[gdf["months_since_launch"] <= 1]["avg_score"]
    launch_score = launch_data.mean() if len(launch_data) else early

    # Classify trajectory
    if late > early + 0.12 and slope > 0:
        trajectory = "RECOVERED"
    elif late < early - 0.08 and slope < 0:
        trajectory = "DECLINING"
    elif overall_avg >= 0.72 and variance < 0.12:
        trajectory = "STABLE_HIGH"
    elif overall_avg < 0.50 and variance < 0.15:
        trajectory = "STABLE_LOW"
    else:
        trajectory = "VOLATILE"

    # Redemption score 0-100
    recovery_delta  = late - early
    redemption_score = round(max(0, min(100,
        (recovery_delta * 50) +
        (overall_avg    * 30) +
        ((1 - variance) * 20)
    )), 1)

    results.append({
        "game_name":        game,
        "trajectory":       trajectory,
        "launch_score":     round(launch_score, 3),
        "early_sentiment":  round(early, 3),
        "late_sentiment":   round(late, 3),
        "sentiment_change": round(late - early, 3),
        "overall_avg":      round(overall_avg, 3),
        "slope":            round(slope, 5),
        "variance":         round(variance, 3),
        "redemption_score": redemption_score,
        "months_of_data":   len(gdf),
        "total_reviews":    int(gdf["review_count"].sum()),
    })

results_df = pd.DataFrame(results).sort_values("redemption_score", ascending=False)

# ── PRINT RESULTS ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SENTIMENT TRAJECTORY CLASSIFICATION")
print("=" * 65)

for traj in ["RECOVERED", "STABLE_HIGH", "VOLATILE", "DECLINING", "STABLE_LOW"]:
    group = results_df[results_df["trajectory"] == traj]
    if len(group) == 0:
        continue
    print(f"\n── {traj} ──")
    for _, row in group.iterrows():
        arrow = "↑" if row["sentiment_change"] > 0 else "↓"
        change = abs(row["sentiment_change"])
        print(f"  {row['game_name']:<26} "
              f"launch: {row['launch_score']:.2f}  "
              f"now: {row['late_sentiment']:.2f}  "
              f"{arrow}{change:.3f}  "
              f"[{row['months_of_data']} months, {row['total_reviews']} reviews]")

print("\n" + "=" * 65)
print("REDEMPTION SCORE LEADERBOARD")
print("=" * 65)
for _, row in results_df.iterrows():
    bar   = "█" * int(row["redemption_score"] / 4)
    traj  = row["trajectory"][:3]
    print(f"  {row['game_name']:<26} {bar:<25} {row['redemption_score']:>5}  [{traj}]")

# ── SAVE ──────────────────────────────────────────────────────────────────
results_df.to_csv("data/processed/trajectory_results.csv", index=False)
monthly.to_csv("data/processed/monthly_sentiment.csv", index=False)
game_reviews.to_csv("data/processed/kaggle_reviews_filtered.csv", index=False)
print("\nSaved:")
print("  data/processed/trajectory_results.csv")
print("  data/processed/monthly_sentiment.csv")
print("  data/processed/kaggle_reviews_filtered.csv")