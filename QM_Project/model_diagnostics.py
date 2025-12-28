# -*- coding: utf-8 -*-
"""
Created on 2025/12/22 00:40

@author: starry
"""



"""
Full pipeline:
1) Read LSOA layer (night transport + IMD already joined)
2) Build model dataframe (ignore population_density for now)
3) Fit OLS (HC3 robust SE)
4) Create a 6-panel diagnostic figure similar to performance::check_model()
5) Save outputs/model_diagnostics.png
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

LSOA_LAYER = "outputs/lsoa_night_transport_venues_summary.geojson"



# Helpers

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute VIF for predictors in X (no y)."""
    X_ = sm.add_constant(X, has_constant="add")
    vifs = []
    for i, col in enumerate(X_.columns):
        if col == "const":
            continue
        vifs.append((col, variance_inflation_factor(X_.values, i)))
    return pd.DataFrame(vifs, columns=["variable", "VIF"]).sort_values("VIF", ascending=False)


def check_model_like(model, model_df: pd.DataFrame, x_cols: list, out_png: str):
    """Create a 6-panel diagnostics figure and save it."""

    # core quantities
    fitted = model.fittedvalues
    resid = model.resid

    infl = model.get_influence()
    stud_resid = infl.resid_studentized_internal
    leverage = infl.hat_matrix_diag
    cooks = infl.cooks_distance[0]

    sqrt_abs_stud = np.sqrt(np.abs(stud_resid))

    # VIF
    X_for_vif = model_df[x_cols].replace([np.inf, -np.inf], np.nan).dropna()
    vif_df = compute_vif(X_for_vif)

    # plotting
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    # 1. Residuals vs Fitted (Linearity)
    ax1.scatter(fitted, resid, s=10, alpha=0.8)
    ax1.axhline(0, linewidth=1)
    lowess_line = sm.nonparametric.lowess(resid, fitted, frac=0.25, it=0)
    ax1.plot(lowess_line[:, 0], lowess_line[:, 1])
    ax1.set_title("Linearity: Residuals vs Fitted")
    ax1.set_xlabel("Fitted values")
    ax1.set_ylabel("Residuals")

    # 2. Scale-Location (Homoscedasticity)
    ax2.scatter(fitted, sqrt_abs_stud, s=10, alpha=0.8)
    lowess_line2 = sm.nonparametric.lowess(sqrt_abs_stud, fitted, frac=0.25, it=0)
    ax2.plot(lowess_line2[:, 0], lowess_line2[:, 1])
    ax2.set_title("Homogeneity of Variance: Scale-Location")
    ax2.set_xlabel("Fitted values")
    ax2.set_ylabel("sqrt(|studentized residuals|)")

    # 3. VIF (Collinearity)
    ax3.barh(vif_df["variable"], vif_df["VIF"])
    ax3.axvline(5, linestyle="--", linewidth=1)
    ax3.axvline(10, linestyle="--", linewidth=1)
    ax3.set_title("Collinearity: VIF (lower is better)")
    ax3.set_xlabel("VIF")
    ax3.invert_yaxis()

    # 4. QQ plot (Normality)
    sm.qqplot(stud_resid, line="45", ax=ax4, marker="o", markersize=3)
    ax4.set_title("Normality of Residuals: QQ plot")
    ax4.set_xlabel("Theoretical Quantiles")
    ax4.set_ylabel("Studentized Residuals")

    # 5. Residual histogram (Normality)
    ax5.hist(stud_resid, bins=40)
    ax5.set_title("Normality of Residuals: Distribution")
    ax5.set_xlabel("Studentized Residuals")
    ax5.set_ylabel("Count")

    # 6. Influence: Residuals vs Leverage (size ~ Cook's distance)
    max_c = np.nanmax(cooks) if np.nanmax(cooks) > 0 else 1.0
    sizes = 50 * (cooks / max_c) + 10
    ax6.scatter(leverage, stud_resid, s=sizes, alpha=0.8)
    ax6.axhline(0, linewidth=1)
    ax6.set_title("Influence: Studentized Residuals vs Leverage\n(point size ~ Cook's distance)")
    ax6.set_xlabel("Leverage (hat values)")
    ax6.set_ylabel("Studentized Residuals")

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.show()

    print(f"[OK] saved diagnostics figure: {out_png}")
    print("\nTop VIFs:")
    print(vif_df.head(10))



# Main

def main():
    if not os.path.exists(LSOA_LAYER):
        raise FileNotFoundError(f"Missing: {LSOA_LAYER}")

    gdf = gpd.read_file(LSOA_LAYER)

    # expected columns
    required = ["IMD_RANK", "night_transport_density_km2"]
    for c in required:
        if c not in gdf.columns:
            raise ValueError(
                f"Column not found: {c}\n"
                f"Available columns: {list(gdf.columns)}"
            )

    # project to BNG (meters)
    gdf = gdf.to_crs(27700)

    # Charing Cross point
    charing_cross = (
        gpd.GeoSeries.from_xy([-0.1246], [51.5079], crs=4326)
        .to_crs(27700)
        .iloc[0]
    )

    # centroid distance (km)
    gdf["centroid"] = gdf.geometry.centroid
    gdf["dist_to_charing_cross_km"] = gdf["centroid"].distance(charing_cross) / 1000

    # clean IMD_RANK (handle commas)
    gdf["imd_rank"] = (
        gdf["IMD_RANK"].astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    gdf["imd_rank"] = pd.to_numeric(gdf["imd_rank"], errors="coerce")

    # predictors (log transform density)
    gdf["night_transport_density_km2"] = pd.to_numeric(gdf["night_transport_density_km2"], errors="coerce")
    gdf["ln_night_transport_density"] = np.log1p(gdf["night_transport_density_km2"])

    # model df
    model_df = gdf[["imd_rank", "ln_night_transport_density", "dist_to_charing_cross_km"]].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

    print("Model sample size (N):", len(model_df))

    # OLS + robust SE
    formula = "imd_rank ~ ln_night_transport_density + dist_to_charing_cross_km"
    model = smf.ols(formula, data=model_df).fit(cov_type="HC3")
    print(model.summary())

    # save regression table
    out_csv = os.path.join(OUT_DIR, "regression_results_imd_rank_night_transport_dist.csv")
    model.summary2().tables[1].to_csv(out_csv)
    print(f"Saved: {out_csv}")

    # diagnostics figure
    out_png = os.path.join(OUT_DIR, "model_diagnostics.png")
    x_cols = ["ln_night_transport_density", "dist_to_charing_cross_km"]
    check_model_like(model, model_df, x_cols=x_cols, out_png=out_png)


if __name__ == "__main__":
    main()
