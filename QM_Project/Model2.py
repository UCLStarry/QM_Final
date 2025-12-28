# -*- coding: utf-8 -*-
"""
Created on 2025/12/26 00:43

@author: starry
"""

"""
OLS mainline closure:
IMD ~ night_transport_density + night_venues_density + venues_to_transport_ratio + population_density + distance_to_central_London

Outputs:
- outputs/model_ols_hc3_summary.txt
- outputs/vif.csv
- outputs/model_diagnostics.png
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.nonparametric.smoothers_lowess import lowess
from shapely.geometry import Point

sns.set_style("whitegrid")

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Prefer GEOJSON
GEO_PATH = "outputs/lsoa_night_transport_venues_summary.geojson"
CSV_PATH = "outputs/lsoa_night_transport_venues_summary.csv"

# 1. Load data

if os.path.exists(GEO_PATH):
    gdf = gpd.read_file(GEO_PATH)
    has_geom = True
elif os.path.exists(CSV_PATH):
    gdf = pd.read_csv(CSV_PATH)
    has_geom = False
else:
    raise FileNotFoundError(f"Cannot find {GEO_PATH} or {CSV_PATH}. Please check your outputs/ folder.")


# 2. Robust column picking (avoid KeyError like IMD_score)

def pick_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

# Key vars (try multiple possible names)
col_imd = pick_col(gdf, ["IMD_score", "IMDScore", "IMD_SCORE", "IMD_RANK", "IMD_rank", "imd_rank", "imd_score"])
col_nt  = pick_col(gdf, ["night_transport_density_km2", "night_transport_density", "nt_density_km2"])
col_nv  = pick_col(gdf, ["night_venues_density_km2", "night_venues_density", "nv_density_km2"])
col_rat = pick_col(gdf, ["venues_to_transport_ratio", "venue_transport_ratio", "ratio"])
col_pop = pick_col(gdf, ["population_density", "POPDEN", "pop_density", "PopDen"])

if col_imd is None:
    raise KeyError("Cannot find an IMD column. Tried: IMD_score/IMDScore/IMD_RANK/imd_rank ...")
if col_nt is None:
    raise KeyError("Cannot find night transport density column (e.g., night_transport_density_km2).")
if col_nv is None:
    raise KeyError("Cannot find night venues density column (e.g., night_venues_density_km2).")
if col_rat is None:
    raise KeyError("Cannot find venues_to_transport_ratio column.")
if col_pop is None:
    raise KeyError("Cannot find population density column (e.g., POPDEN or population_density).")

print("Using columns:")
print("  IMD:", col_imd)
print("  night_transport_density:", col_nt)
print("  night_venues_density:", col_nv)
print("  venues_to_transport_ratio:", col_rat)
print("  population_density:", col_pop)

# 3. Compute distance_to_central_London if possible

dist_col = pick_col(gdf, ["distance_to_central_London", "distance_to_city_centre", "dist_central_km"])

if dist_col is None:
    if not has_geom:
        raise ValueError(
            "No distance column found AND you loaded CSV (no geometry). "
            "Use the GeoJSON version to compute distance."
        )

    # Charing Cross (WGS84)
    charing_cross = Point(-0.1276, 51.5074)
    cc_gdf = gpd.GeoDataFrame(geometry=[charing_cross], crs="EPSG:4326")

    # Ensure CRS
    if gdf.crs is None:

        gdf = gdf.set_crs(epsg=4326)

    # Project to British National Grid
    gdf_proj = gdf.to_crs(epsg=27700)
    cc_proj = cc_gdf.to_crs(epsg=27700)

    # centroid distance (km)
    gdf_proj["distance_to_central_London"] = gdf_proj.centroid.distance(cc_proj.iloc[0].geometry) / 1000.0

    # attach back
    gdf["distance_to_central_London"] = gdf_proj["distance_to_central_London"].values
    dist_col = "distance_to_central_London"
    print("Computed distance_to_central_London (km) from centroids to Charing Cross.")


# 4. Build model dataframe + transforms

df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore")).copy()

# Coerce numeric
for c in [col_imd, col_nt, col_nv, col_rat, col_pop, dist_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

imd_is_rank = ("rank" in col_imd.lower())

df["y_log_imd"] = np.log(df[col_imd]) if (imd_is_rank) else np.log1p(df[col_imd])
df["x_log_nt"]  = np.log1p(df[col_nt])
df["x_log_nv"]  = np.log1p(df[col_nv])
df["x_log_rat"] = np.log1p(df[col_rat])
df["x_log_pop"] = np.log1p(df[col_pop])
df["x_dist"]    = df[dist_col]  # already km

model_df = df[["y_log_imd", "x_log_nt", "x_log_nv", "x_log_rat", "x_log_pop", "x_dist"]].dropna().copy()
print("Model N:", len(model_df))


# 5. OLS + HC3 robust

formula = "y_log_imd ~ x_log_nt + x_log_nv + x_log_rat + x_log_pop + x_dist"
ols = smf.ols(formula=formula, data=model_df).fit(cov_type="HC3")

# Save summary
with open(os.path.join(OUT_DIR, "model_ols_hc3_summary.txt"), "w", encoding="utf-8") as f:
    f.write(ols.summary().as_text())

print(ols.summary())


# 6. VIF

X = model_df[["x_log_nt", "x_log_nv", "x_log_rat", "x_log_pop", "x_dist"]].copy()
X_const = sm.add_constant(X)

vif_rows = []
for i, col in enumerate(X_const.columns):
    if col == "const":
        continue
    vif = variance_inflation_factor(X_const.values, i)
    vif_rows.append({"variable": col, "VIF": float(vif)})

vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
vif_df.to_csv(os.path.join(OUT_DIR, "vif.csv"), index=False)
print("\nVIF:")
print(vif_df)


# 7. Diagnostics figure (similar spirit to check_model)

influence = ols.get_influence()
fitted = ols.fittedvalues.values
resid = ols.resid.values
stud_resid = influence.resid_studentized_internal  # standardized residuals

# Panel A: Linearity (residuals vs fitted + LOWESS)
low_a = lowess(resid, fitted, frac=0.3, return_sorted=True)

# Panel B: Homogeneity of variance (sqrt(|std resid|) vs fitted + LOWESS)
sqrt_abs = np.sqrt(np.abs(stud_resid))
low_b = lowess(sqrt_abs, fitted, frac=0.3, return_sorted=True)

# Panel C: VIF dot plot (log scale like check_model does conceptually)
vif_plot = vif_df.copy()
vif_plot["log_vif"] = np.log(vif_plot["VIF"].clip(lower=1e-6))

# Panel D: Normality QQ
pp = sm.ProbPlot(stud_resid)

# Extra Panel E: Residual density
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, :])

# A Linearity
ax1.scatter(fitted, resid, alpha=0.35)
ax1.plot(low_a[:, 0], low_a[:, 1], linewidth=2)
ax1.axhline(0, linestyle="--", linewidth=1)
ax1.set_title("Linearity\nReference line should be flat and horizontal", loc="left", fontsize=12)
ax1.set_xlabel("Fitted values")
ax1.set_ylabel("Residuals")

# B Homogeneity
ax2.scatter(fitted, sqrt_abs, alpha=0.35)
ax2.plot(low_b[:, 0], low_b[:, 1], linewidth=2)
ax2.set_title("Homogeneity of Variance\nReference line should be flat and horizontal", loc="left", fontsize=12)
ax2.set_xlabel("Fitted values")
ax2.set_ylabel(r"$\sqrt{|Std.\ residuals|}$")

# C Collinearity (VIF)
ax3.scatter(vif_plot["variable"], vif_plot["VIF"])
ax3.axhspan(0, 5, alpha=0.15)
ax3.axhspan(5, 10, alpha=0.10)
ax3.axhspan(10, max(12, vif_plot["VIF"].max() * 1.05), alpha=0.06)
ax3.set_title("Collinearity\nHigh collinearity (VIF) may inflate parameter uncertainty", loc="left", fontsize=12)
ax3.set_xlabel("")
ax3.set_ylabel("Variance Inflation Factor (VIF)")
ax3.tick_params(axis="x", rotation=20)

# D QQ plot
pp.qqplot(line="45", ax=ax4)
ax4.set_title("Normality of Residuals\nDots should fall along the line", loc="left", fontsize=12)
ax4.set_xlabel("Standard Normal Distribution Quantiles")
ax4.set_ylabel("Sample Quantile Deviations")

# E Residual density
sns.kdeplot(resid, ax=ax5, fill=True)
ax5.set_title("Normality of Residuals\nDistribution should be close to the normal curve", loc="left", fontsize=12)
ax5.set_xlabel("Residuals")
ax5.set_ylabel("Density")

plt.tight_layout()
diag_path = os.path.join(OUT_DIR, "model_diagnostics1.png")
plt.savefig(diag_path, dpi=200, bbox_inches="tight")
plt.show()

print("\nSaved:")
print(" -", os.path.join(OUT_DIR, "model_ols_hc3_summary.txt"))
print(" -", os.path.join(OUT_DIR, "vif.csv"))
print(" -", diag_path)

