# -*- coding: utf-8 -*-
"""
Created on 2025/12/22 22:26

@author: starry
"""

# -*- coding: utf-8 -*-
"""
KMeans clustering for night venues + night transport + IMD at LSOA level
Outputs:
  - outputs/pictures/kmeans_elbow.png
  - outputs/pictures/kmeans_scatter.png
  - outputs/pictures/kmeans_clusters_map.png
  - outputs/lsoa_venue_clusters.geojson

Inputs:
  - outputs/lsoa_night_transport_venues_summary.csv
  - outputs/lsoa_night_transport_venues_summary.geojson
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

OUT_DIR = "../outputs"
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUT_DIR, "lsoa_night_transport_venues_summary.csv")
GEOJSON_PATH = os.path.join(OUT_DIR, "lsoa_night_transport_venues_summary.geojson")


# Load data

df = pd.read_csv(CSV_PATH)
gdf = gpd.read_file(GEOJSON_PATH)


lsoa_id_candidates = ["LSOA11CD"]
LSOA_ID = next((c for c in lsoa_id_candidates if c in df.columns), None)
if LSOA_ID is None:
    raise ValueError(f"Cannot find LSOA id field in CSV. Candidates tried: {lsoa_id_candidates}")


# 1. Choose clustering features (minimal viable + strong story)


feature_candidates = {
    "night_venues_density_km2": ["night_venues_density_km2"],
    "venues_to_transport_ratio": ["venues_to_transport_ratio"],
    "night_transport_density_km2": ["night_transport_density_km2"],
    "imd": ["IMD_RANK"]
}

def pick_col(cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

col_venues_den = pick_col(feature_candidates["night_venues_density_km2"])
col_ratio      = pick_col(feature_candidates["venues_to_transport_ratio"])
col_transport  = pick_col(feature_candidates["night_transport_density_km2"])
col_imd        = pick_col(feature_candidates["imd"])

required = [col_venues_den, col_ratio, col_transport, col_imd]
if any(c is None for c in required):
    raise ValueError(
        "Missing one or more required columns for clustering.\n"
        f"Found: venues_den={col_venues_den}, ratio={col_ratio}, transport={col_transport}, imd={col_imd}\n"
        f"CSV columns: {list(df.columns)}"
    )

features = [col_venues_den, col_ratio, col_transport, col_imd]

# dropna for clustering rows only
df_model = df[[LSOA_ID] + features].dropna().copy()


# 2. Standardize

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model[features].values)


# 3. Figure 1: Elbow plot

inertias = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.title("K-means: Elbow plot (Night venues + transport + IMD, LSOA)")
plt.xticks(list(k_values))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "kmeans_elbow.png"), dpi=200)
plt.show()


# 4. Fit final KMeans (choose k)

k_val = 5
kmeans = KMeans(n_clusters=k_val, random_state=42, n_init="auto")
df_model["cluster_kmeans"] = kmeans.fit_predict(X_scaled)

# merge cluster back to df
df = df.merge(df_model[[LSOA_ID, "cluster_kmeans"]], on=LSOA_ID, how="left")


# 5. Figure 2: 2D scatter

# venues_density vs transport_density or ratio vs imd
x_var = col_venues_den
y_var = col_transport

plot_df = df_model.copy()  # only rows used in clustering

plt.figure(figsize=(7, 5))
sc = plt.scatter(
    plot_df[x_var],
    plot_df[y_var],
    c=plot_df["cluster_kmeans"],
    cmap="tab10",
    alpha=0.7,
    edgecolor="k",
    linewidth=0.2
)
plt.xlabel(x_var)
plt.ylabel(y_var)
#plt.title(f"K-means clusters (k={k_val}) on {x_var} vs {y_var} (LSOA)")
plt.title(f"K-means clusters (k={k_val})(LSOA)")
handles, labels = sc.legend_elements(prop="colors", alpha=0.7)
labels = [f"Cluster {i}" for i in range(k_val)]
plt.legend(handles, labels, title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "kmeans_scatter.png"), dpi=200)
plt.show()

# 6. Figure 3: Map of clusters (LSOA polygons)

from matplotlib.patches import Patch

# Join cluster labels onto GeoDataFrame
LSOA_ID_G = next((c for c in lsoa_id_candidates if c in gdf.columns), None)
if LSOA_ID_G is None:
    raise ValueError(f"Cannot find LSOA id field in GEOJSON. Candidates tried: {lsoa_id_candidates}")

gdf_out = gdf.merge(df[[LSOA_ID, "cluster_kmeans"]], left_on=LSOA_ID_G, right_on=LSOA_ID, how="left")

# Ensure projected CRS for clean plotting
try:
    gdf_out = gdf_out.to_crs(epsg=27700)
except Exception:
    pass

fig, ax = plt.subplots(figsize=(8, 8))

# tab10 颜色（与前面 scatter 保持一致：Cluster 0..4 对应 tab10(0..4)）
cmap = plt.get_cmap("tab10")

# 只取地图中真实出现过的 cluster（避免图例出现多余类别）
present_clusters = sorted(gdf_out["cluster_kmeans"].dropna().astype(int).unique())

# 先画缺失值（No cluster）
gdf_out[gdf_out["cluster_kmeans"].isna()].plot(
    ax=ax,
    color="lightgrey",
    linewidth=0.15,
    edgecolor="white"
)

# 再逐个 cluster 画（固定颜色）
for c in present_clusters:
    gdf_out[gdf_out["cluster_kmeans"] == c].plot(
        ax=ax,
        color=cmap(c),
        linewidth=0.15,
        edgecolor="white"
    )

#  No cluster）
handles = [Patch(facecolor=cmap(c), edgecolor="none", label=f"Cluster {c}") for c in present_clusters]
handles.append(Patch(facecolor="lightgrey", edgecolor="none", label="No cluster"))

ax.legend(
    handles=handles,
    title="Cluster",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=True
)

ax.set_title(f"LSOA K-means clusters (k={k_val}): Night venues + transport + IMD")
ax.set_aspect("equal", "box")

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "kmeans_clusters_map.png"), dpi=200)
plt.show()



# 7. Export clustered GeoJSON

out_geojson = os.path.join(OUT_DIR, "lsoa_venue_clusters.geojson")
gdf_out.drop(columns=[LSOA_ID], errors="ignore").to_file(out_geojson, driver="GeoJSON")

print("Saved:")
print(" - outputs/pictures/kmeans_elbow.png")
print(" - outputs/pictures/kmeans_scatter.png")
print(" - outputs/pictures/kmeans_clusters_map.png")
print(" - outputs/lsoa_venue_clusters.geojson")

cluster_summary = (
    df.groupby("cluster_kmeans")[features]
      .mean()
      .round(2)
)

cluster_summary.to_csv("outputs/cluster_summary.csv")
print("Cluster summary saved to outputs/cluster_summary.csv")


