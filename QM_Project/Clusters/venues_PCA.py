# -*- coding: utf-8 -*-
"""
Created on 2025/12/22 22:59

@author: starry
"""


"""
PCA scatter plots for LSOA night-time economy variables (loop version)
- Produces multiple PCA scatter plots, each colored by a different continuous variable
- Robust to IMD column name variations and duplicate columns after merges

Input:
  - outputs/lsoa_night_transport_venues_summary.csv

Outputs:
  - outputs/pca_scatter_colored_by_<var>.png   (one per var)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Paths

OUT_DIR = "../outputs"
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "lsoa_night_transport_venues_summary.csv")


# Load data

df = pd.read_csv(CSV_PATH)


# Helpers

def pick_col(df_, candidates):
    """Pick the first column that exists in df_ from a list of candidates."""
    for c in candidates:
        if c in df_.columns:
            return c
    return None

def get_1d_series(df_, colname):
    """
    Force a single 1D Series even if duplicate column names exist
    (in that case pandas may return a DataFrame).
    """
    obj = df_[colname]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj


# Identify required columns

col_venues = pick_col(df, ["night_venues_density_km2"])
col_transport = pick_col(df, ["night_transport_density_km2"])
col_ratio = pick_col(df, ["venues_to_transport_ratio"])
col_imd = pick_col(df, ["IMD_RANK", "IMD_RANK_x", "IMD_RANK_y", "imd_rank", "IMDRank", "imd"])

pca_features = [col_venues, col_transport, col_ratio, col_imd]

if any(c is None for c in pca_features):
    raise ValueError(
        "Missing one or more PCA feature columns.\n"
        f"Detected: venues={col_venues}, transport={col_transport}, ratio={col_ratio}, imd={col_imd}\n"
        f"Available columns: {list(df.columns)}"
    )


# Build feature DataFrame (1D safe) and drop NA rows

feature_df = pd.DataFrame({
    col_venues: get_1d_series(df, col_venues),
    col_transport: get_1d_series(df, col_transport),
    col_ratio: get_1d_series(df, col_ratio),
    col_imd: get_1d_series(df, col_imd),
})

mask_features = feature_df.notna().all(axis=1)
feature_df = feature_df.loc[mask_features].copy()


# Standardize + PCA

X_scaled = StandardScaler().fit_transform(feature_df.values)
pca = PCA()
pca_np = pca.fit_transform(X_scaled)

# Create PCA DataFrame (PC1, PC2, ...)
df_pca = pd.DataFrame(
    pca_np,
    columns=[f"PC{i}" for i in range(1, pca_np.shape[1] + 1)]
)


# Choose variables to color by (loop)

color_vars = [
    col_imd,       # inequality dimension (higher rank = less deprived)
    col_venues,    # night-time economy intensity
    col_ratio,     # supply-demand mismatch proxy
    col_transport  # transport provision
]


# Produce one plot per color variable

for color_var in color_vars:
    # Build a 1D color series aligned to feature_df rows (same index)
    color_series = get_1d_series(df.loc[feature_df.index], color_var)

    # Drop NA for this color var (keep PCA coordinates aligned)
    mask = color_series.notna()
    df_plot = df_pca.loc[mask.values].copy()
    df_plot[color_var] = color_series.loc[mask].values

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        df_plot["PC1"],
        df_plot["PC2"],
        c=df_plot[color_var],
        cmap="viridis",
        alpha=0.8,
        edgecolor="none"
    )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)")
    plt.title(f"PCA Scatter Plot Colored by {color_var}")
    plt.colorbar(scatter, label=color_var)
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, f"pca_scatter_colored_by_{color_var}.png")
    plt.savefig(out_png, dpi=200)
    plt.show()

    print(f"Saved: {out_png}")

print("\nExplained variance ratio (first 4 PCs):")
for i, r in enumerate(pca.explained_variance_ratio_[:4], start=1):
    print(f"  PC{i}: {r:.4f}")

print(f"\nPCA features used: {pca_features}")
print(f"Rows used for PCA: {len(feature_df)}")
print(f"Color vars plotted: {color_vars}")
