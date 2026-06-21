# Copyright (c) 2026 Mohibul Hoque
# Licensed under the MIT License (see LICENSE file for details)
# Author: Mohibul Hoque <explorarhok@gmail.com> (github.com/speedyhok)

import requests
import pandas as pd
import time

GAMES = {
    "Cyberpunk 2077":        1091500,
    "No Man's Sky":          275850,
    "Elden Ring":            1245620,
    "Battlefield 2042":      1517290,
    "Hogwarts Legacy":       990080,
    "Starfield":             1716740,
    "Fallout 76":            1151340,
    "Hades":                 1145360,
    "Baldurs Gate 3":        1086940,
    "Stardew Valley":        413150,
    "The Last of Us Part I": 1888930,
    "Disco Elysium":         632470,
    "Dying Light 2":         534380,
    "Gotham Knights":        1496790,
    "Forspoken":             1680630,
    "Red Dead Redemption 2": 1174180,
    "Counter-Strike 2":      730,
    "GTA V":                 271590,
    "Hollow Knight":         367520,
    "Dead by Daylight":      381210,
}

MAX_REVIEWS = 2000


def get_reviews(appid, game_name):
    print(f"\n── {game_name} ──")
    reviews = []
    cursor  = "*"
    retries = 0

    while len(reviews) < MAX_REVIEWS:
        url = f"https://store.steampowered.com/appreviews/{appid}"
        params = {
            "json":                    1,
            "num_per_page":            100,
            "language":                "english",
            "cursor":                  cursor,
            "filter":                  "recent",
            "purchase_type":           "all",
            "review_type":             "all",
            "filter_offtopic_activity": 0,
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            data     = response.json()
        except Exception as e:
            print(f"  Error: {e}")
            retries += 1
            if retries > 3:
                print(f"  Giving up after 3 retries")
                break
            time.sleep(10)
            continue

        batch = data.get("reviews", [])
        if not batch:
            print(f"  No more reviews available")
            break

        for r in batch:
            reviews.append({
                "game_name":              game_name,
                "appid":                  appid,
                "review_id":              r["recommendationid"],
                "posted_date":            pd.to_datetime(r["timestamp_created"], unit="s"),
                "is_positive":            r["voted_up"],
                "playtime_at_review_hrs": round(r["author"]["playtime_at_review"] / 60, 2),
                "playtime_forever_hrs":   round(r["author"]["playtime_forever"] / 60, 2),
                "steam_purchase":         r["steam_purchase"],
                "received_for_free":      r["received_for_free"],
                "num_reviews_by_author":  r["author"]["num_reviews"],
                "num_games_owned":        r["author"]["num_games_owned"],
                "helpfulness_score":      r["weighted_vote_score"],
                "review_text":            r["review"],
            })

        new_cursor = data.get("cursor")
        if not new_cursor or new_cursor == cursor:
            print(f"  Cursor stopped — no more pages")
            break

        cursor  = new_cursor
        retries = 0

        collected = len(reviews)
        print(f"  Collected {collected} / {MAX_REVIEWS}", end="\r")
        time.sleep(1.5)

    # Trim to exactly 2000 and deduplicate
    df = pd.DataFrame(reviews).drop_duplicates(subset="review_id").head(MAX_REVIEWS)
    print(f"  Done — {len(df)} reviews saved")
    return df


# ── MAIN ──────────────────────────────────────────────────────────────────
all_dfs = []

for game_name, appid in GAMES.items():
    df = get_reviews(appid, game_name)
    all_dfs.append(df)

    filename = game_name.lower().replace(" ", "_").replace("'", "").replace(" ", "_")
    df.to_csv(f"data/raw/{filename}_reviews.csv", index=False)

    time.sleep(2)

# ── SAVE COMBINED ─────────────────────────────────────────────────────────
final_df = pd.concat(all_dfs, ignore_index=True)
final_df.to_csv("data/raw/all_reviews.csv", index=False)

print(f"\n{'='*50}")
print(f"COLLECTION COMPLETE")
print(f"{'='*50}")
print(f"Total reviews collected: {len(final_df)}")
print(f"Games collected:         {final_df['game_name'].nunique()}")
print(f"Date range:              {final_df['posted_date'].min()} to {final_df['posted_date'].max()}")
print(f"\nReviews per game:")
print(final_df["game_name"].value_counts().to_string())