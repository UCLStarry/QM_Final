# -*- coding: utf-8 -*-
"""
Created on 2025/12/26 00:35

@author: starry
"""

"""
Mainline model closure:
IMD ~ night_transport_density + night_venues_density + ratio + population_density + distance_to_central_London

Outputs:
- outputs/model_summary.txt
- outputs/vif.csv
- outputs/model_diagnostics.png
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

from shapely.geometry import Point
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_style("whitegrid")


# Config

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Prefer GeoJSON because we can compute distance via geometry
DATA_PATH = "outputs/lsoa_night_transport_venues_summary.geojson"

# Charing Cross (WGS84)
CHARING_CROSS_LONLAT = (-0.1276, 51.5074)


# Helpers

def read_data(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".geojson", ".json", ".gpkg", ".shp"]:
        gdf = gpd.read_file(path)
        return gdf, True
    elif ext == ".csv":
        df = pd.read_csv(path)
        return df, False
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def ensure_distance_to_central_london(gdf: gpd.GeoDataFrame, out_col="distance_to_central_London") -> gpd.GeoDataFrame:
    """
    Compute centroid distance (km) from each polygon to Charing Cross.
    Requires geometry in GeoDataFrame.
    """
    if out_col in gdf.columns and gdf[out_col].notna().any():
        return gdf

    if gdf.geometry is None:
        raise ValueError("No geometry found; cannot compute distance_to_central_London.")

    # Ensure CRS
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    # Build Charing Cross point
    cc = gpd.GeoDataFrame(
        geometry=[Point(*CHARING_CROSS_LONLAT)],
        crs="EPSG:4326"
    )

    # Project to British National Grid for metric distances
    gdf_proj = gdf.to_crs(epsg=27700)
    cc_proj = cc.to_crs(epsg=27700)

    # Centroid distance in km
    gdf[out_col] = gdf_proj.centroid.distance(cc_proj.iloc[0].geometry) / 1000.0
    return gdf

def pick_first_existing(df: pd.DataFrame, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of these columns exist: {candidates}")
    return None

def safe_log1p(s: pd.Series):
    return np.log1p(pd.to_numeric(s, errors="coerce"))

def compute_vif(design_df: pd.DataFrame) -> pd.DataFrame:
    """
    VIF requires numeric matrix with intercept excluded usually.
    We'll compute VIF on predictors only (no intercept).
    """
    X = design_df.copy()
    X = X.dropna()

    # Drop constant columns
    nunique = X.nunique(dropna=True)
    X = X.loc[:, nunique > 1]

    vif_rows = []
    X_np = X.values
    for i, col in enumerate(X.columns):
        vif_rows.append({"variable": col, "VIF": float(variance_inflation_factor(X_np, i))})
    return pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)


# 1. Read data

data, is_geo = read_data(DATA_PATH)

if is_geo:
    gdf = data
    gdf = ensure_distance_to_central_london(gdf, out_col="distance_to_central_London")
else:
    # CSV: you must already have distance_to_central_London (or centroid lon/lat) in the table
    gdf = data
    if "distance_to_central_London" not in gdf.columns:
        raise ValueError(
            "You loaded CSV but it has no geometry, so I cannot compute distance_to_central_London.\n"
            "Either (1) switch to the GeoJSON path, or (2) add centroid lon/lat columns, or (3) precompute distance."
        )


# 2. Column mapping (auto-detect)

# Outcome (IMD): you used IMD_RANK before. Keep flexible.
col_imd = pick_first_existing(gdf, ["IMD_RANK", "IMD_rank", "imd_rank", "IMDScore", "IMD_score"])

col_transport = pick_first_existing(gdf, [
    "night_transport_density_km2", "night_transport_density", "night_transport_density_km^2"
])

col_venues = pick_first_existing(gdf, [
    "night_venues_density_km2", "night_venues_density"
])

col_ratio = pick_first_existing(gdf, [
    "venues_to_transport_ratio", "venue_to_transport_ratio", "transport_venue_ratio"
])

col_popden = pick_first_existing(gdf, [
    "POPDEN", "population_density", "pop_density", "PopDensity"
])

col_dist = pick_first_existing(gdf, [
    "distance_to_central_London", "distance_to_city_centre", "distance_to_centre"
])


# 3. Build model dataframe (clean + transforms)

df = pd.DataFrame({
    "IMD": pd.to_numeric(gdf[col_imd], errors="coerce"),
    "night_transport_density": pd.to_numeric(gdf[col_transport], errors="coerce"),
    "night_venues_density": pd.to_numeric(gdf[col_venues], errors="coerce"),
    "ratio": pd.to_numeric(gdf[col_ratio], errors="coerce"),
    "population_density": pd.to_numeric(gdf[col_popden], errors="coerce"),
    "distance": pd.to_numeric(gdf[col_dist], errors="coerce"),
})

# Practical cleaning
df = df.replace([np.inf, -np.inf], np.nan).dropna()


df["log_transport"] = safe_log1p(df["night_transport_density"])
df["log_venues"] = safe_log1p(df["night_venues_density"])
df["log_ratio"] = safe_log1p(df["ratio"])
df["log_popden"] = safe_log1p(df["population_density"])
df["log_distance"] = safe_log1p(df["distance"])

# Outcome transform:
# Option A: keep raw IMD (recommended if it's IMD Score)
# Option B: log1p(IMD) (recommended if skewed and positive)

df["y"] = safe_log1p(df["IMD"])


# 4. Fit OLS with HC3 robust SE

formula = "y ~ log_transport + log_venues + log_ratio + log_popden + log_distance"
model = smf.ols(formula=formula, data=df).fit(cov_type="HC3")

# Save text summary
summary_path = os.path.join(OUT_DIR, "model_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(model.summary().as_text())

print("Model N =", int(model.nobs))
print("Saved:", summary_path)

# 5. VIF

# Build design matrix (predictors only) from the transformed columns
X_for_vif = df[["log_transport", "log_venues", "log_ratio", "log_popden", "log_distance"]].copy()
vif_df = compute_vif(X_for_vif)
vif_path = os.path.join(OUT_DIR, "vif.csv")
vif_df.to_csv(vif_path, index=False)
print("Saved:", vif_path)
print(vif_df)


# 6. Diagnostics plot (6-panel)

fitted = model.fittedvalues
resid = model.resid
influence = model.get_influence()
std_resid = influence.resid_studentized_internal
cooks = influence.cooks_distance[0]
leverage = influence.hat_matrix_diag

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

# (1) Residuals vs Fitted
axes[0].scatter(fitted, resid, alpha=0.35)
axes[0].axhline(0, linewidth=1)
axes[0].set_title("Residuals vs Fitted")
axes[0].set_xlabel("Fitted values")
axes[0].set_ylabel("Residuals")

# (2) Normal Q-Q
sm.qqplot(std_resid, line="45", ax=axes[1])
axes[1].set_title("Normal Q-Q (Studentized Residuals)")

# (3) Scale-Location: sqrt(|std resid|) vs fitted
axes[2].scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.35)
axes[2].set_title("Scale-Location")
axes[2].set_xlabel("Fitted values")
axes[2].set_ylabel("sqrt(|studentized residual|)")

# (4) Residual histogram
axes[3].hist(resid, bins=40)
axes[3].set_title("Residual Distribution")
axes[3].set_xlabel("Residual")
axes[3].set_ylabel("Count")

# (5) Leverage vs Studentized Residuals (with Cook's distance size)
# scale cooks for size (avoid huge)
size = 30 + 500 * np.clip(cooks, 0, np.quantile(cooks, 0.95))
axes[4].scatter(leverage, std_resid, s=size, alpha=0.35)
axes[4].axhline(0, linewidth=1)
axes[4].set_title("Leverage vs Studentized Residuals\n(bubble size ~ Cook's D)")
axes[4].set_xlabel("Leverage")
axes[4].set_ylabel("Studentized residual")

# (6) Cook's distance
axes[5].stem(cooks, linefmt="-", markerfmt=" ", basefmt=" ")
axes[5].set_title("Cook's Distance")
axes[5].set_xlabel("Observation index")
axes[5].set_ylabel("Cook's D")

plt.tight_layout()
diag_path = os.path.join(OUT_DIR, "model_diagnostics.png")
plt.savefig(diag_path, dpi=200)
plt.show()

print("Saved:", diag_path)

# 7. Optional: quick interpretation printout

coefs = model.params.round(4)
pvals = model.pvalues
print("\nKey coefficients (log-log-ish interpretation):")
for k in ["log_transport", "log_venues", "log_ratio", "log_popden", "log_distance"]:
    if k in coefs.index:
        print(f"  {k:12s} coef={coefs[k]:>8}  p={pvals[k]:.3g}")
