# Video Game Review Manipulation Detector

**Author:** Mohibul Hoque  
**Email:** explorarhok@gmail.com  
**Tools:** Python · Pandas · scikit-learn · Tableau · Steam Web API · Kaggle  
**Dataset Size:** 113,000+ reviews across 19 major game launches  
**Timeline:** 2024  

---

## Project Overview

This project investigates whether data can distinguish between coordinated review manipulation and genuine organic backlash in video game launches. Most review analysis tools stop at flagging suspicious accounts. This project goes further — classifying each game's full reputation arc from launch to present, identifying the strongest behavioural signals of manipulation, and surfacing insights unavailable in any public review tool.

The central question:

> *Which game launches show statistically abnormal review behaviour — and can we tell the difference between coordinated manipulation and genuine organic backlash?*

---

## Repository Structure

```
review-manipulation-detector/
│
├── data/
│   ├── raw/
│   │   ├── steam_reviews/          # Individual game CSVs from Steam API
│   │   ├── games.csv               # Kaggle Metacritic game metadata
│   │   ├── games_reviews.csv       # Kaggle historical reviews (3.8M rows)
│   │   └── feature_importance.csv  # Manual ML feature importance data
│   │
│   └── processed/
│       ├── all_reviews_clean.csv       # 38,000 Steam reviews with engineered features
│       ├── daily_sentiment.csv         # Day-by-day sentiment with spike flags
│       ├── monthly_sentiment.csv       # Monthly aggregated sentiment per game
│       ├── ml_results.csv              # All 5 model predictions per review
│       ├── trajectory_results.csv      # Trajectory classification per game
│       ├── kaggle_reviews_filtered.csv # 78,000 filtered Kaggle reviews
│       └── master_summary.csv          # 19 games x 25 columns — Tableau input
│
├── scripts/
│   ├── 01_collect_reviews.py       # Steam API collection with rate limiting
│   ├── 02_feature_engineering.py   # Suspicion scoring and feature creation
│   ├── 03_daily_sentiment.py       # Daily aggregation and spike detection
│   ├── 04_ml_models.py             # All 5 ML models and agreement scoring
│   ├── 05_sentiment_decay.py       # Trajectory classification and redemption score
│   └── 06_master_summary.py        # Final join of all data sources
│
└── README.md
```

---

## Data Sources

### Source 1 — Steam Web API
- **Volume:** 38,000 reviews across 19 games (2,000 per game)
- **Cost:** Free, no API key required
- **Purpose:** Behavioural manipulation signals
- **Key fields:** playtime at review, account age, games owned, free copy flag, helpfulness score
- **Limitation:** Returns recent reviews only — no historical date filtering available through the free API

### Source 2 — Kaggle Metacritic Dataset
- **Volume:** 78,120 filtered user reviews from 3.8 million total
- **Coverage:** Launch day to March 2025, up to 103 months per game
- **Purpose:** Historical sentiment trajectory analysis
- **Key fields:** review date, score (0-100), review type (user/critic), game metadata
- **URL:** kaggle.com/datasets

### Why Two Sources

The Steam API provides rich behavioural data — playtime, account history, purchase verification — but cannot retrieve reviews by date. The Kaggle dataset provides full historical coverage going back to launch day but lacks behavioural signals. The two sources are complementary: Steam data answers *who* is reviewing suspiciously, Kaggle data answers *how* sentiment evolved over time.

---

## Target Games

| Game | Trajectory | Launch Score | Current Score | Critic-User Gap |
|---|---|---|---|---|
| No Man's Sky | RECOVERED | 0.37 | 0.69 | 16 pts |
| Red Dead Redemption 2 | RECOVERED | 0.78 | 0.92 | 8 pts |
| The Last of Us Part I | RECOVERED | 0.58 | 0.86 | 13 pts |
| Cyberpunk 2077 | RECOVERED | 0.70 | 0.87 | 14 pts |
| Battlefield 2042 | RECOVERED | 0.17 | 0.39 | 45 pts |
| Gotham Knights | RECOVERED | 0.48 | 0.63 | 14 pts |
| Dying Light 2 | RECOVERED | 0.60 | 0.71 | 4 pts |
| Counter-Strike 2 | RECOVERED | 0.35 | 0.41 | 31 pts |
| Elden Ring | STABLE_HIGH | 0.74 | 0.86 | 14 pts |
| Hollow Knight | STABLE_HIGH | 0.88 | 0.88 | -1 pts |
| Stardew Valley | STABLE_HIGH | 0.89 | 0.90 | 1 pt |
| Baldur's Gate 3 | STABLE_HIGH | 0.94 | 0.87 | 4 pts |
| Hogwarts Legacy | STABLE_HIGH | 0.78 | 0.82 | 2 pts |
| Disco Elysium | STABLE_HIGH | 0.75 | 0.86 | 6 pts |
| Fallout 76 | VOLATILE | 0.29 | 0.35 | 23 pts |
| Starfield | VOLATILE | 0.64 | 0.65 | 15 pts |
| Dead by Daylight | VOLATILE | 0.51 | 0.66 | 9 pts |
| Forspoken | VOLATILE | 0.47 | 0.65 | 22 pts |
| Hades | DECLINING | 0.89 | 0.78 | 7 pts |

---

## Methodology

### Phase 1 — Data Collection (Script 01)

Steam reviews were collected using the public appreviews endpoint with the following parameters:

- Filter: recent (most recent 2,000 reviews per game)
- Language: English only
- Purchase type: all (including non-Steam purchases)
- Rate limiting: 1.5 second delay between requests, 3 retry attempts on failure

The Kaggle dataset was filtered to the 19 target games using Metacritic game IDs, retaining user reviews only with valid scores.

### Phase 2 — Feature Engineering (Script 02)

Eight behavioural features were engineered from raw Steam review data:

| Feature | Definition | Signal |
|---|---|---|
| is_low_playtime | Playtime at review < 1 hour | Reviewing without playing |
| playtime_ratio | Playtime at review / total playtime | Reviewed and never returned |
| is_first_review | Author has written exactly 1 review ever | Throwaway account |
| is_thin_account | Author owns fewer than 5 games | Manipulation account |
| review_hour | Hour of day review was posted | Bot activity patterns |
| days_since_launch | Days between launch date and review date | Launch window behaviour |
| is_launch_window | Review posted within first 7 days | Highest manipulation period |
| helpfulness_score | Steam community helpfulness rating | Community trust signal |

These features were combined into a composite suspicion score (0-100):

```
suspicion_score = (is_low_playtime x 25) + (is_first_review x 20) +
                  (is_thin_account x 20) + (received_for_free x 15) +
                  ((1 - helpfulness_score) x 20)
```

Reviews with suspicion_score >= 40 were labelled is_suspicious = 1 for supervised model training.

### Phase 3 — Spike Detection (Script 03)

A rolling average spike detector was built to identify review bomb days:

- Calculate 7-day rolling average of negative reviews per game
- Flag any day where negative reviews exceed 3x that rolling average
- Additional condition: sentiment ratio must drop below 0.5 on the same day
- This eliminates false positives from positive review spikes

Result: 12 confirmed negative spike days detected across 19 games.

### Phase 4 — Machine Learning (Script 04)

Five models were applied across two paradigms:

**Unsupervised Models (no labels required)**

| Model | Purpose | Result |
|---|---|---|
| K-Means (k=3) | Behavioural segmentation | Found 592 reviews in suspicious cluster — characterised by under 30 minutes playtime from established accounts |
| Isolation Forest | Anomaly detection | 1,898 anomalies detected — Fallout 76 (347) and Gotham Knights (243) highest |
| DBSCAN | Density-based outlier detection | 847 outliers across 32 natural clusters — largely consistent with Isolation Forest |

**Supervised Models (trained on suspicion score labels)**

| Model | Accuracy | ROC-AUC | Notes |
|---|---|---|---|
| Random Forest | 98.2% | 0.978 | Trained on 4 continuous features only to avoid circular validation |
| Logistic Regression | 66.4% | 0.952 | High recall (96%) prioritised — better to investigate false positives than miss real ones |

A critical methodological note: an initial version of the supervised models achieved 100% accuracy because the boolean feature flags used to build the suspicion score label were included as training features. This circular validation was identified and corrected by restricting supervised models to continuous features only (playtime, review count, games owned, helpfulness score).

**Model Agreement Score**

Each review received a score from 0-5 counting how many models flagged it as suspicious. Reviews flagged by all 5 models simultaneously represent the highest confidence manipulation signals:

- All 5 models agreed: 9 reviews total
- Fallout 76: 3, Battlefield 2042: 2, Gotham Knights: 2, Elden Ring: 1, Hogwarts Legacy: 1

### Phase 5 — Sentiment Trajectory Analysis (Script 05)

Monthly sentiment ratios were calculated from the Kaggle historical dataset and a linear regression slope was fitted per game. Games were classified using the following rules:

| Class | Condition |
|---|---|
| RECOVERED | Late sentiment > early sentiment + 0.12 and positive slope |
| DECLINING | Late sentiment < early sentiment - 0.08 and negative slope |
| STABLE_HIGH | Overall average >= 0.72 and variance < 0.12 |
| STABLE_LOW | Overall average < 0.50 and variance < 0.15 |
| VOLATILE | None of the above |

A Redemption Score (0-100) was calculated for each game:

```
redemption_score = (recovery_delta x 50) + (overall_avg x 30) + ((1 - variance) x 20)
```

### Phase 6 — Master Summary (Script 06)

All processed data was joined into a single 19 x 25 master table combining Steam manipulation signals, Kaggle trajectory classification, Metacritic scores, and the Redemption Score. This file is the primary Tableau data source.

---

## Key Findings

**Finding 1 — Cyberpunk 2077's backlash was entirely genuine**

Only 4 suspicious reviews detected out of 2,000 collected. Near-zero suspicious activity across all 5 models. The anger at launch was real — players were legitimately furious about a broken product. The subsequent recovery to 0.87 sentiment reflects genuine improvement, not reputation management.

**Finding 2 — Battlefield 2042 has the most manipulated review profile in the dataset**

142 K-Means suspicious reviews, 222 Isolation Forest anomalies, 2 reviews flagged by all 5 models, a 45-point critic-user score gap, and a launch sentiment of 0.17 — the lowest in the dataset. Every signal points the same direction.

**Finding 3 — Free review copies are the primary manipulation vector**

Random Forest identified received_for_free as the strongest feature at 52% importance — more predictive than zero playtime or throwaway accounts, which are commonly assumed to be the main signals. This suggests gifted review copies and giveaway campaigns are more commonly exploited than bot networks.

**Finding 4 — Hades is silently declining**

Launched at 0.89 sentiment — second highest in the dataset — now at 0.78 after 53 months. The only game classified as DECLINING. No public tool is currently reporting this trend. Likely driven by the Epic Games Store exclusivity period and a long wait between releases.

**Finding 5 — Hollow Knight users rated it higher than critics**

Score gap of -1.0 — the only game in the dataset where the user score exceeds the critic score. 95 months of data showing near-zero sentiment variance. Genuinely undervalued at launch by professional reviewers.

**Finding 6 — No Man's Sky is the greatest recovery story in modern gaming — proven by data**

Sentiment moved from 0.37 at launch to 0.69 current across 93 months and 2,092 reviews. A +0.324 change over 8 years is the largest recovery delta in the dataset. This is no longer just a narrative — it is measurable.

---

## Tableau Dashboard

The project findings are presented across 4 dashboard pages in Tableau:

**Page 1 — The Landscape**
Full overview of all 19 games. Scatter plot of launch vs current sentiment, Redemption Score leaderboard, and 4 KPI summary cards.

**Page 2 — The Recovery Room**
Sentiment recovery curves for all RECOVERED games overlaid on a single timeline chart. Side by side comparison of recovery speed and magnitude.

**Page 3 — The Manipulation Lab**
K-Means suspicious review counts, Isolation Forest anomaly counts, model agreement chart, and feature importance comparison between Random Forest and Logistic Regression.

**Page 4 — Critics vs The People**
Dumbbell chart comparing Metascore vs user score per game, score gap bar chart ranked by disagreement magnitude, and scatter plot with trend line showing systematic critic-user divergence.

---

## Limitations and Known Issues

**Data limitations**

- Steam API does not support date-based filtering. All Steam reviews are recent only — launch window manipulation data, which would be the most valuable, is not captured.
- Two separate data sources rather than one unified dataset creates a join dependency on game name matching.
- Forspoken has no Steam data — the game had insufficient English reviews available at collection time.
- GTA V is present in Steam data but absent from the Kaggle trajectory dataset — no sentiment arc available for that title.
- Counter-Strike 2 has only 17 months of Kaggle data — trajectory classification is less reliable than for older games.

**Machine learning limitations**

- No real ground truth labels exist. The is_suspicious label was self-defined using the suspicion score formula, which means supervised model validation is partially circular even after the continuous-feature fix.
- DBSCAN outliers turned out to characterise power users with very high playtime and review counts rather than manipulation accounts — the outlier signal measures statistical unusualness, not necessarily suspicious behaviour.
- Isolation Forest contamination parameter (0.05) was set manually. Different contamination values produce meaningfully different anomaly counts.
- K-Means requires the number of clusters to be specified upfront. The choice of k=3 was informed but not empirically validated.

**Dashboard limitations**

- No cross-dashboard filter actions implemented. Clicking a game on one chart does not highlight it across other charts.
- No navigation buttons linking the 4 dashboards into a continuous story flow.
- Fixed dashboard sizing may cause some charts to be cut off on smaller screens.
- Some parameter reference line labels remain visible on charts — minor cosmetic issue.

---

## Installation and Setup

**Requirements**

```
Python 3.8+
pandas
requests
scikit-learn
beautifulsoup4
pytrends
```

**Install dependencies**

```bash
pip install requests pandas scikit-learn beautifulsoup4 pytrends
```

**Run the pipeline in order**

```bash
python scripts/01_collect_reviews.py
python scripts/02_feature_engineering.py
python scripts/03_daily_sentiment.py
python scripts/04_ml_models.py
python scripts/05_sentiment_decay.py
python scripts/06_master_summary.py
```

Note: Script 01 takes approximately 10-15 minutes to collect 38,000 reviews. Steam rate limiting is handled automatically. Script 05 takes 1-2 minutes to load the 3.8 million row Kaggle file.

**Kaggle data**

Download the following datasets and place both CSV files in `data/raw/`:
- games.csv
- games_reviews.csv

---

## Future Improvements

- Integrate SteamDB changelog scraping to annotate review spikes with patch release dates — distinguishing patch-triggered backlash from unexplained spikes
- Add Google Trends data correlation to identify viral controversy moments
- Implement a regret reviewer signal — accounts that changed their review from positive to negative after 100+ hours, which cannot be faked by bots
- Build a Streamlit web application so the analysis can be run interactively without Tableau
- Extend to 50+ games to improve ML model reliability and trajectory classification robustness
- Add cross-dashboard filter actions in Tableau for interactive game-level drill-down

---

## As a bonus I added original Tableau file also , Do check it out.

## Screenshots from Tableau dashboard

<img width="1358" height="708" alt="Gr1" src="https://github.com/user-attachments/assets/d77bf709-16e5-43df-8819-85468287cd57" />
<img width="1323" height="716" alt="gr2" src="https://github.com/user-attachments/assets/6f90e451-ac32-4bc3-9b88-8a9dff21d18f" />
<img width="1202" height="708" alt="gr3" src="https://github.com/user-attachments/assets/13d211ad-46a9-4550-9146-6beb929cf5c9" />
<img width="1291" height="748" alt="Gr4" src="https://github.com/user-attachments/assets/a6bfa00a-76bf-412b-827b-dbeb719cb38f" />

## Contact

**Mohibul Hoque**  
explorarhok@gmail.com  

For questions about methodology, data sources, or to suggest improvements, feel free to reach out by email.
