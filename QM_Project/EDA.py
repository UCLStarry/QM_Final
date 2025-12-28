# -*- coding: utf-8 -*-
"""
Created on 2025/12/26 00:16

@author: starry
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import Point

sns.set_style("whitegrid")

GEO_PATH = "outputs/lsoa_night_transport_venues_summary.geojson"

#1.Load LSOA
gdf = gpd.read_file(GEO_PATH)

#2.Compute distance_to_central_London（Charing Cross）
# Charing Cross (WGS84)
charing_cross = Point(-0.1276, 51.5074)
charing_cross_gdf = gpd.GeoDataFrame(
    geometry=[charing_cross], crs="EPSG:4326"
)


# Ensure that LSOA is WGS84
if gdf.crs is None:
    gdf = gdf.set_crs(epsg=4326)

# Project to British National Grid
gdf_proj = gdf.to_crs(epsg=27700)
cc_proj = charing_cross_gdf.to_crs(epsg=27700)

# Calculate the distance (km) from LSOA centroid to the center.
gdf_proj["distance_to_central_London"] = (
    gdf_proj.centroid.distance(cc_proj.iloc[0].geometry) / 1000
)

# Restore to original gdf
gdf["distance_to_central_London"] = gdf_proj["distance_to_central_London"]

numeric_vars = [
    "night_transport_density_km2",
    "night_venues_density_km2",
    "venues_to_transport_ratio",
    "IMD_RANK",
    "POPDEN",
    "distance_to_central_London"
]

numeric_df = gdf[numeric_vars].dropna()
numeric_long = numeric_df.melt(var_name="variable", value_name="value")

vars_unique = numeric_long["variable"].unique()
cols = 3
rows = (len(vars_unique) + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
axes = axes.flatten()

for i, var in enumerate(vars_unique):
    sns.histplot(
        data=numeric_long[numeric_long["variable"] == var],
        x="value",
        bins=30,
        color="steelblue",
        ax=axes[i]
    )
    axes[i].set_title(var)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Count")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("Distributions of Night-time Economy, Transport and Deprivation Variables", y=1.02)
sns.despine()
plt.show()



gdf["log_IMD"] = np.log(gdf["IMD_RANK"])
gdf["log_night_transport_density"] = np.log(gdf["night_transport_density_km2"] + 1)
gdf["log_night_venues_density"] = np.log(gdf["night_venues_density_km2"] + 1)

long_df = pd.melt(
    gdf,
    id_vars=["log_IMD"],
    value_vars=[
        "log_night_transport_density",
        "log_night_venues_density",
        "venues_to_transport_ratio"
    ],
    var_name="predictor",
    value_name="x_value"
)

label_map = {
    "log_night_transport_density": "log(Night-time Transport Density)",
    "log_night_venues_density": "log(Night-time Venue Density)",
    "venues_to_transport_ratio": "Venues-to-Transport Ratio"
}

predictors = long_df["predictor"].unique()
cols = 2
rows = (len(predictors) + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
axes = axes.flatten()

for i, p in enumerate(predictors):
    subset = long_df[long_df["predictor"] == p]

    sns.regplot(
        data=subset,
        x="x_value",
        y="log_IMD",
        scatter_kws={"alpha": 0.4, "color": "steelblue"},
        line_kws={"color": "black"},
        ax=axes[i]
    )

    axes[i].set_title(f"log(IMD) vs {label_map[p]}")
    axes[i].set_xlabel(label_map[p])
    axes[i].set_ylabel("log(IMD)")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("Bivariate Relationships Between Night-time Provision and Deprivation", y=1.02)
sns.despine()
plt.show()
