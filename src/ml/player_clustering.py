"""
Player Clustering — Group players into archetypes using unsupervised learning.

Models: K-Means, Hierarchical (Agglomerative).
"""

import sys
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import ML_MODEL_DIR, RANDOM_STATE


CLUSTER_FEATURES = [
    "avg_runs", "avg_sr", "total_fours", "total_sixes",
    "avg_pp_runs", "avg_mid_runs", "avg_death_runs",
    "avg_wickets", "avg_economy", "avg_dots_bowled",
    "total_wickets", "avg_pp_wkts", "avg_death_wkts",
]

ARCHETYPE_LABELS = {
    "high_sr_high_runs": "Power Hitter / Aggressor",
    "low_sr_high_runs": "Anchor / Accumulator",
    "death_specialist_bat": "Finisher",
    "high_wickets_low_econ": "Strike Bowler",
    "low_wickets_low_econ": "Economical Bowler",
    "allrounder": "All-Rounder",
}


def cluster_players(player_df: pd.DataFrame, n_clusters: int = 5) -> dict:
    """Cluster players and assign archetype labels."""
    df = player_df.copy()

    available_features = [c for c in CLUSTER_FEATURES if c in df.columns]
    if len(available_features) < 3:
        print("❌ Not enough features for clustering.")
        return {}

    X = df[available_features].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    print(f"\n🔄 Running K-Means (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    km_labels = kmeans.fit_predict(X_scaled)
    km_silhouette = silhouette_score(X_scaled, km_labels)
    print(f"  ✅ K-Means silhouette score: {km_silhouette:.4f}")

    # Hierarchical
    print(f"\n🔄 Running Agglomerative Clustering (k={n_clusters})...")
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg.fit_predict(X_scaled)
    agg_silhouette = silhouette_score(X_scaled, agg_labels)
    print(f"  ✅ Agglomerative silhouette score: {agg_silhouette:.4f}")

    # Use K-Means as primary
    df["cluster"] = km_labels

    # PCA for visualization
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    df["pca_1"] = X_pca[:, 0]
    df["pca_2"] = X_pca[:, 1]

    # Label clusters based on centroids
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=available_features,
    )

    cluster_labels = {}
    for i in range(n_clusters):
        c = centroids.iloc[i]
        if c.get("avg_runs", 0) > centroids["avg_runs"].median() and c.get("avg_sr", 0) > centroids["avg_sr"].median():
            label = "Power Hitter"
        elif c.get("avg_runs", 0) > centroids["avg_runs"].median():
            label = "Anchor"
        elif c.get("total_wickets", 0) > centroids["total_wickets"].median() and c.get("avg_economy", 0) < centroids["avg_economy"].median():
            label = "Strike Bowler"
        elif c.get("avg_economy", 0) < centroids["avg_economy"].median():
            label = "Economical Bowler"
        elif c.get("avg_death_runs", 0) > centroids["avg_death_runs"].median():
            label = "Finisher"
        else:
            label = "All-Rounder"
        cluster_labels[i] = label

    df["archetype"] = df["cluster"].map(cluster_labels)

    # Optimal k analysis
    inertias = []
    silhouettes = []
    k_range = range(2, min(11, len(df) // 2))
    for k in k_range:
        km_temp = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels_temp = km_temp.fit_predict(X_scaled)
        inertias.append(km_temp.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels_temp))

    # Save model
    ML_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(ML_MODEL_DIR / "player_clustering_model.pkl", "wb") as f:
        pickle.dump({
            "kmeans": kmeans,
            "scaler": scaler,
            "pca": pca,
            "features": available_features,
            "cluster_labels": cluster_labels,
        }, f)

    results = {
        "clustered_df": df,
        "kmeans": kmeans,
        "km_silhouette": km_silhouette,
        "agg_silhouette": agg_silhouette,
        "centroids": centroids,
        "cluster_labels": cluster_labels,
        "pca": pca,
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
        "features": available_features,
    }

    print(f"\n✅ Clustering complete. {n_clusters} clusters assigned.")
    return results


if __name__ == "__main__":
    from src.warehouse.schema import get_connection
    from src.ml.feature_engineering import build_player_features

    conn = get_connection()
    players = build_player_features(conn, source="t20i")
    conn.close()

    if not players.empty:
        results = cluster_players(players, n_clusters=5)
        if results:
            print("\nCluster distribution:")
            print(results["clustered_df"]["archetype"].value_counts())
