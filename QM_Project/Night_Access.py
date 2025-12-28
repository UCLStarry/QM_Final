# -*- coding: utf-8 -*-
"""
Created on 2025/12/21 22:21

@author: starry
"""

import geopandas as gpd
import pandas as pd

LSOA_PATH = "LSOA.shp"
NIGHT_BUS_STOPS = "night_bus_stops.geojson"
NIGHT_TUBE_STATIONS = "night_tube_stations.geojson"

OUT_CSV = "lsoa_night_access.csv"
OUT_GEOJSON = "lsoa_night_access.geojson"

# 1. read
lsoa = gpd.read_file(LSOA_PATH).to_crs("EPSG:27700")  # British National Grid for robust spatial joins
bus = gpd.read_file(NIGHT_BUS_STOPS).to_crs("EPSG:27700")
tube = gpd.read_file(NIGHT_TUBE_STATIONS).to_crs("EPSG:27700")

# 2. spatial join: points within polygons
bus_join = gpd.sjoin(bus[["stop_id","geometry"]], lsoa[["LSOA11CD","geometry"]], how="inner", predicate="within")
tube_join = gpd.sjoin(tube[["station_id","geometry"]], lsoa[["LSOA11CD","geometry"]], how="inner", predicate="within")

bus_ct = bus_join.groupby("LSOA11CD")["stop_id"].nunique().rename("night_bus_stop_count")
tube_ct = tube_join.groupby("LSOA11CD")["station_id"].nunique().rename("night_tube_station_count")

# 3. merge back
out = lsoa.merge(bus_ct, on="LSOA11CD", how="left").merge(tube_ct, on="LSOA11CD", how="left")
out["night_bus_stop_count"] = out["night_bus_stop_count"].fillna(0).astype(int)
out["night_tube_station_count"] = out["night_tube_station_count"].fillna(0).astype(int)

# Simple composite indicator (you can refine later: z-score, log, weights)
out["night_transit_score"] = out["night_bus_stop_count"] + 3 * out["night_tube_station_count"]

# 4. export
out.drop(columns="geometry").to_csv(OUT_CSV, index=False, encoding="utf-8")
out.to_file(OUT_GEOJSON, driver="GeoJSON")
print(f"Saved -> {OUT_CSV} and {OUT_GEOJSON}")
