# Copyright (c) 2026 Mohibul Hoque
# Licensed under the MIT License (see LICENSE file for details)
# Author: Mohibul Hoque <explorarhok@gmail.com> (github.com/speedyhok)

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score)
import warnings
warnings.filterwarnings("ignore")

# ── LOAD DATA ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/all_reviews_clean.csv")
print(f"Loaded {len(df)} reviews\n")

# Full features for unsupervised models
features = [
    "playtime_at_review_hrs",
    "num_reviews_by_author",
    "num_games_owned",
    "helpfulness_score",
    "is_low_playtime",
    "is_first_review",
    "is_thin_account",
    "received_for_free"
]

# Reduced features for supervised models
# Removes boolean flags that were used to build the suspicion score label
# Forces models to find patterns in raw continuous data only
supervised_features = [
    "playtime_at_review_hrs",
    "num_reviews_by_author",
    "num_games_owned",
    "helpfulness_score"
]

ml_df = df[features + ["game_name", "suspicion_score"]].dropna().copy()

# Threshold at 40 for better class balance
ml_df["is_suspicious"] = (ml_df["suspicion_score"] >= 40).astype(int)

# Full feature matrix for unsupervised models
X = ml_df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduced feature matrix for supervised models
X_supervised = ml_df[supervised_features]
scaler_sup = StandardScaler()
X_scaled_supervised = scaler_sup.fit_transform(X_supervised)

print(f"Running models on {len(ml_df)} reviews")
print(f"Suspicious reviews (label=1): {ml_df['is_suspicious'].sum()} "
      f"({ml_df['is_suspicious'].mean()*100:.1f}%)")
print(f"Genuine reviews   (label=0): {(ml_df['is_suspicious']==0).sum()}\n")

# ══════════════════════════════════════════════════════════════════════════
# MODEL 1: K-MEANS CLUSTERING (Unsupervised)
# Uses full feature set — no labels needed
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("MODEL 1: K-Means Clustering (Unsupervised)")
print("=" * 60)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
ml_df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

# Name clusters by playtime — lowest playtime = most suspicious
cluster_means = ml_df.groupby("kmeans_cluster")["playtime_at_review_hrs"].mean()
sorted_clusters = cluster_means.sort_values()
kmeans_names = {
    sorted_clusters.index[0]: "Suspicious",
    sorted_clusters.index[1]: "Bot-like",
    sorted_clusters.index[2]: "Genuine"
}
ml_df["kmeans_label"] = ml_df["kmeans_cluster"].map(kmeans_names)

print("\nCluster averages:")
print(ml_df.groupby("kmeans_label")[features].mean().round(2).to_string())
print("\nCluster sizes:")
print(ml_df["kmeans_label"].value_counts().to_string())
print("\nCluster breakdown by game:")
print(ml_df.groupby(["game_name", "kmeans_label"]).size()
      .unstack(fill_value=0).to_string())

# ══════════════════════════════════════════════════════════════════════════
# MODEL 2: ISOLATION FOREST (Unsupervised Anomaly Detection)
# Uses full feature set — finds outliers without labels
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 2: Isolation Forest (Anomaly Detection)")
print("=" * 60)

iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
ml_df["isolation_pred"] = iso.fit_predict(X_scaled)
ml_df["isolation_score"] = iso.score_samples(X_scaled)

ml_df["isolation_label"] = ml_df["isolation_pred"].map(
    {-1: "Anomaly", 1: "Normal"}
)

print("\nAnomaly detection results:")
print(ml_df["isolation_label"].value_counts().to_string())

print("\nAnomalies per game:")
anomalies = ml_df[ml_df["isolation_label"] == "Anomaly"]
print(anomalies["game_name"].value_counts().to_string())

print("\nAverage features — Anomaly vs Normal:")
print(ml_df.groupby("isolation_label")[features].mean().round(2).to_string())

agreement = (
    (ml_df["isolation_label"] == "Anomaly") ==
    (ml_df["is_suspicious"] == 1)
).mean()
print(f"\nAgreement with suspicion score labels: {agreement*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# MODEL 3: DBSCAN (Unsupervised Density-Based Outlier Detection)
# Uses full feature set — finds outliers that don't fit any cluster
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 3: DBSCAN (Density-Based Outlier Detection)")
print("=" * 60)

db = DBSCAN(eps=0.5, min_samples=5)
ml_df["dbscan_cluster"] = db.fit_predict(X_scaled)

ml_df["dbscan_label"] = ml_df["dbscan_cluster"].apply(
    lambda x: "Outlier" if x == -1 else f"Cluster {x}"
)
ml_df["is_dbscan_outlier"] = (ml_df["dbscan_cluster"] == -1).astype(int)

n_clusters = len(set(ml_df["dbscan_cluster"])) - (
    1 if -1 in ml_df["dbscan_cluster"].values else 0
)
n_outliers = (ml_df["dbscan_cluster"] == -1).sum()

print(f"\nClusters found: {n_clusters}")
print(f"Outliers found: {n_outliers} ({n_outliers/len(ml_df)*100:.1f}%)")

print("\nOutliers per game:")
outliers = ml_df[ml_df["dbscan_label"] == "Outlier"]
print(outliers["game_name"].value_counts().to_string())

print("\nAverage features — Outliers vs Core points:")
print(ml_df.groupby("is_dbscan_outlier")[features].mean().round(2).to_string())

db_agreement = (
    (ml_df["dbscan_label"] == "Outlier") ==
    (ml_df["is_suspicious"] == 1)
).mean()
print(f"\nAgreement with suspicion score labels: {db_agreement*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# MODEL 4: RANDOM FOREST (Supervised — class balanced)
# Uses REDUCED feature set only — no circular boolean flags
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 4: Random Forest (Supervised — reduced features)")
print("=" * 60)
print(f"Training on: {supervised_features}")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_supervised, ml_df["is_suspicious"],
    test_size=0.2, random_state=42,
    stratify=ml_df["is_suspicious"]
)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print(f"\nAccuracy:      {accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.3f}")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  True Genuine:    {cm[0][0]}  |  False Suspicious: {cm[0][1]}")
print(f"  False Genuine:   {cm[1][0]}  |  True Suspicious:  {cm[1][1]}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Genuine", "Suspicious"]))

print("\nFeature Importance:")
importance = pd.Series(rf.feature_importances_, index=supervised_features)
importance = importance.sort_values(ascending=False)
for feat, score in importance.items():
    bar = "█" * int(score * 50)
    print(f"  {feat:<30} {bar} {score:.3f}")

ml_df["rf_suspicious_prob"] = rf.predict_proba(X_scaled_supervised)[:, 1]
ml_df["rf_label"] = rf.predict(X_scaled_supervised)
ml_df["rf_label"] = ml_df["rf_label"].map({0: "Genuine", 1: "Suspicious"})

# ══════════════════════════════════════════════════════════════════════════
# MODEL 5: LOGISTIC REGRESSION (Supervised — probability scoring)
# Uses REDUCED feature set only — no circular boolean flags
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 5: Logistic Regression (Probability Scoring)")
print("=" * 60)
print(f"Training on: {supervised_features}")

lr = LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000
)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_prob = lr.predict_proba(X_test)[:, 1]

print(f"\nAccuracy:      {accuracy_score(y_test, lr_pred)*100:.1f}%")
print(f"ROC-AUC Score: {roc_auc_score(y_test, lr_prob):.3f}")
print("\nConfusion Matrix:")
cm_lr = confusion_matrix(y_test, lr_pred)
print(f"  True Genuine:    {cm_lr[0][0]}  |  False Suspicious: {cm_lr[0][1]}")
print(f"  False Genuine:   {cm_lr[1][0]}  |  True Suspicious:  {cm_lr[1][1]}")
print("\nClassification Report:")
print(classification_report(y_test, lr_pred,
      target_names=["Genuine", "Suspicious"]))

print("\nFeature Coefficients (higher = stronger suspicious signal):")
coef = pd.Series(np.abs(lr.coef_[0]), index=supervised_features)
coef = coef.sort_values(ascending=False)
for feat, score in coef.items():
    bar = "█" * int(score * 10)
    print(f"  {feat:<30} {bar} {score:.3f}")

ml_df["lr_suspicious_prob"] = lr.predict_proba(X_scaled_supervised)[:, 1]
ml_df["lr_label"] = lr.predict(X_scaled_supervised)
ml_df["lr_label"] = ml_df["lr_label"].map({0: "Genuine", 1: "Suspicious"})

# ══════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON — How Much Do All 5 Models Agree?
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL AGREEMENT COMPARISON")
print("=" * 60)

ml_df["models_flagging"] = (
    (ml_df["isolation_label"]  == "Anomaly").astype(int) +
    (ml_df["is_dbscan_outlier"]                        ) +
    (ml_df["rf_label"]         == "Suspicious").astype(int) +
    (ml_df["lr_label"]         == "Suspicious").astype(int) +
    (ml_df["kmeans_label"]     == "Suspicious").astype(int)
)

high_conf = ml_df[ml_df["models_flagging"] == 5]
print(f"\nHigh confidence (flagged by ALL 5 models): {len(high_conf)}")
print(high_conf["game_name"].value_counts().to_string())

med_conf = ml_df[ml_df["models_flagging"] >= 3]
print(f"\nMedium confidence (flagged by 3+ models): {len(med_conf)}")
print(med_conf["game_name"].value_counts().to_string())

print("\nAverage models agreeing per game:")
print(ml_df.groupby("game_name")["models_flagging"]
      .mean().round(2)
      .sort_values(ascending=False).to_string())

print("\nModel agreement distribution:")
print(ml_df["models_flagging"].value_counts().sort_index().to_string())

# ══════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════
output_cols = [
    "game_name", "suspicion_score", "is_suspicious",
    "kmeans_label",
    "isolation_label", "isolation_score",
    "dbscan_label", "is_dbscan_outlier",
    "rf_label", "rf_suspicious_prob",
    "lr_label", "lr_suspicious_prob",
    "models_flagging"
] + features

ml_df[output_cols].to_csv("data/processed/ml_results.csv", index=False)
print(f"\nSaved ml_results.csv")
print(f"Shape: {ml_df.shape[0]} reviews × {len(output_cols)} columns")