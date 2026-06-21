# Copyright (c) 2026 Mohibul Hoque
# Licensed under the MIT License (see LICENSE file for details)
# Author: Mohibul Hoque <explorarhok@gmail.com> (github.com/speedyhok)

import pandas as pd

print("Building master summary table...")

# ── LOAD ALL PROCESSED DATA ───────────────────────────────────────────────
steam_df      = pd.read_csv("data/processed/all_reviews_clean.csv")
trajectory_df = pd.read_csv("data/processed/trajectory_results.csv")
kaggle_games  = pd.read_csv("data/raw/games.csv")

# ── STEAM SUMMARY PER GAME ────────────────────────────────────────────────
steam_summary = steam_df.groupby("game_name").agg(
    total_steam_reviews     = ("review_id", "count"),
    avg_suspicion_score     = ("suspicion_score", "mean"),
    pct_low_playtime        = ("is_low_playtime", "mean"),
    pct_first_review        = ("is_first_review", "mean"),
    pct_thin_account        = ("is_thin_account", "mean"),
    pct_free_copy           = ("received_for_free", "mean"),
    avg_playtime_at_review  = ("playtime_at_review_hrs", "mean"),
    pct_positive_steam      = ("is_positive", "mean"),
).reset_index()

steam_summary["avg_suspicion_score"] = steam_summary["avg_suspicion_score"].round(1)
steam_summary["pct_low_playtime"]    = (steam_summary["pct_low_playtime"] * 100).round(1)
steam_summary["pct_first_review"]    = (steam_summary["pct_first_review"] * 100).round(1)
steam_summary["pct_thin_account"]    = (steam_summary["pct_thin_account"] * 100).round(1)
steam_summary["pct_free_copy"]       = (steam_summary["pct_free_copy"] * 100).round(1)
steam_summary["avg_playtime_at_review"] = steam_summary["avg_playtime_at_review"].round(1)
steam_summary["pct_positive_steam"]  = (steam_summary["pct_positive_steam"] * 100).round(1)

# ── KAGGLE GAME METADATA ──────────────────────────────────────────────────
TARGET_IDS = {
    1300144696: "Cyberpunk 2077",
    1300223342: "No Man's Sky",
    1300501979: "Elden Ring",
    1300538028: "Battlefield 2042",
    1300523817: "Hogwarts Legacy",
    1300486989: "Starfield",
    1300486337: "Fallout 76",
    1300494289: "Hades",
    1300501848: "Baldurs Gate 3",
    1300170113: "Stardew Valley",
    1300572186: "The Last of Us Part I",
    1300468042: "Disco Elysium",
    1300486966: "Dying Light 2",
    1300522308: "Gotham Knights",
    1300518494: "Forspoken",
    1300461445: "Red Dead Redemption 2",
    1300602226: "Counter-Strike 2",
    1300450663: "Hollow Knight",
    1300454843: "Dead by Daylight",
}

kaggle_filtered = kaggle_games[kaggle_games["id"].isin(TARGET_IDS.keys())].copy()
kaggle_filtered["game_name"] = kaggle_filtered["id"].map(TARGET_IDS)

meta = kaggle_filtered[[
    "game_name", "metascore", "userscore",
    "releaseDate", "developer", "genres"
]].copy()

# Score gap: positive = critics liked more, negative = users liked more
meta["userscore_normalized"] = meta["userscore"]
meta["score_gap"] = (meta["metascore"] - meta["userscore_normalized"]).round(1)

# ── JOIN EVERYTHING ───────────────────────────────────────────────────────
master = trajectory_df.merge(steam_summary, on="game_name", how="left")
master = master.merge(meta, on="game_name", how="left")

# ── MANIPULATION RISK FLAG ────────────────────────────────────────────────
# High suspicion + low playtime + lots of thin accounts = manipulation risk
master["manipulation_risk"] = "LOW"
master.loc[
    (master["avg_suspicion_score"] > 30) |
    (master["pct_low_playtime"] > 40) |
    (master["pct_thin_account"] > 20),
    "manipulation_risk"
] = "MEDIUM"
master.loc[
    (master["avg_suspicion_score"] > 45) &
    (master["pct_low_playtime"] > 50),
    "manipulation_risk"
] = "HIGH"

# ── FINAL COLUMN ORDER ────────────────────────────────────────────────────
cols = [
    "game_name", "releaseDate", "developer", "genres",
    "trajectory", "redemption_score",
    "launch_score", "early_sentiment", "late_sentiment", "sentiment_change",
    "overall_avg", "months_of_data", "total_reviews",
    "metascore", "userscore_normalized", "score_gap",
    "avg_suspicion_score", "manipulation_risk",
    "pct_low_playtime", "pct_first_review", "pct_thin_account", "pct_free_copy",
    "avg_playtime_at_review", "pct_positive_steam", "total_steam_reviews",
]

master = master[cols].sort_values("redemption_score", ascending=False)

# ── PRINT SUMMARY ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MASTER SUMMARY TABLE")
print("=" * 70)
print(master[[
    "game_name", "trajectory", "redemption_score",
    "launch_score", "late_sentiment", "metascore",
    "userscore_normalized", "score_gap", "manipulation_risk"
]].to_string(index=False))

print("\n" + "=" * 70)
print("MANIPULATION RISK BREAKDOWN")
print("=" * 70)
print(master[["game_name", "avg_suspicion_score", "pct_low_playtime",
              "pct_thin_account", "pct_free_copy", "manipulation_risk"
              ]].sort_values("avg_suspicion_score", ascending=False).to_string(index=False))

print("\n" + "=" * 70)
print("CRITIC VS USER SCORE GAP")
print("=" * 70)
print(master[["game_name", "metascore", "userscore_normalized", "score_gap"
              ]].sort_values("score_gap", ascending=False).to_string(index=False))

# ── SAVE ──────────────────────────────────────────────────────────────────
master.to_csv("data/processed/master_summary.csv", index=False)
print("\nSaved: data/processed/master_summary.csv")
print(f"Shape: {master.shape[0]} games × {master.shape[1]} columns")