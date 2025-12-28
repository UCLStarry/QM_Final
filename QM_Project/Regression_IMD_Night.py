# -*- coding: utf-8 -*-
"""
Created on 2025/12/21 23:49

@author: starry
"""


# -*- coding: utf-8 -*-
"""
Regression: IMD ~ night transport density + distance to city centre
(Temporarily ignore population_density because LSOA 2021/2011 lookup not ready)
"""

import numpy as np
import geopandas as gpd
import statsmodels.formula.api as smf


# 1. Read LSOA layer (already has night transport + IMD)

LSOA_LAYER = "outputs/lsoa_night_transport_venues_summary.geojson"
gdf = gpd.read_file(LSOA_LAYER)


# 2. Column mapping (based on your actual file columns)

colmap = {
    "lsoa": "LSOA11CD",
    "imd": "IMD_RANK",  # main DV (continuous rank)
    "night_tr_den": "night_transport_density_km2",
}

for k, v in colmap.items():
    if v not in gdf.columns:
        raise ValueError(
            f"Column not found: {v}\n"
            f"Available columns: {list(gdf.columns)}"
        )


# 3. Geometry prep: project to BNG, compute distance to Charing Cross

gdf = gdf.to_crs(27700)

# Charing Cross (WGS84 -> BNG)
charing_cross = (
    gpd.GeoSeries.from_xy([-0.1246], [51.5079], crs=4326)
    .to_crs(27700)
    .iloc[0]
)

# centroid distance (km)
gdf["centroid"] = gdf.geometry.centroid
gdf["dist_to_charing_cross_km"] = gdf["centroid"].distance(charing_cross) / 1000


# 4. Build model dataframe

df = gdf.copy()

# clean IMD rank (handle commas/strings just in case)
df["imd_rank"] = (
    df[colmap["imd"]].astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["imd_rank"] = np.where(df["imd_rank"].isin(["nan", "None", ""]), np.nan, df["imd_rank"])
df["imd_rank"] = df["imd_rank"].astype(float)

# log transform night transport density
df["night_transport_density_km2"] = df[colmap["night_tr_den"]]
df["ln_night_transport_density"] = np.log1p(df["night_transport_density_km2"])

model_df = df[
    ["imd_rank", "ln_night_transport_density", "dist_to_charing_cross_km"]
].replace([np.inf, -np.inf], np.nan).dropna()

print("Model sample size (N):", len(model_df))


# 5. Regression: OLS + robust SE (HC3)

formula = """
imd_rank
~ ln_night_transport_density
+ dist_to_charing_cross_km
"""

model = smf.ols(formula, data=model_df).fit(cov_type="HC3")
print(model.summary())


# 6. Save regression table

out = model.summary2().tables[1]
out.to_csv("outputs/regression_results_imd_rank_night_transport_dist.csv")
print("Saved: outputs/regression_results_imd_rank_night_transport_dist.csv")


