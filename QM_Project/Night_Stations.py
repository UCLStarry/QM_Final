# -*- coding: utf-8 -*-
"""
Created on 2025/12/21 22:34

@author: starry
"""

import geopandas as gpd
import pandas as pd


# 1. Load data

bus = gpd.read_file("night_bus_stops.geojson")
tube = gpd.read_file("night_tube_stations.geojson")

print("Bus columns:", list(bus.columns))
print("Tube columns:", list(tube.columns))


# 2. Night Bus: Field Standardization

bus_std = bus.rename(columns={
    "stop_id": "stop_id",
    "stop_name": "stop_name",
    "line_id": "line_id"
})

bus_std["mode"] = "night_bus"

bus_std = bus_std[[
    "stop_id",
    "stop_name",
    "line_id",
    "mode",
    "geometry"
]]


# 3. Night Tube：Field Standardization

tube_std = tube.rename(columns={
    "station_id": "stop_id",
    "station_name": "stop_name",
    "line_id": "line_id"
})

tube_std["mode"] = "night_tube"

tube_std = tube_std[[
    "stop_id",
    "stop_name",
    "line_id",
    "mode",
    "geometry"
]]


# 4. CRS Unification

if bus_std.crs != tube_std.crs:
    tube_std = tube_std.to_crs(bus_std.crs)


# 5. Merge into one night transportation point layer

night_transport = pd.concat(
    [bus_std, tube_std],
    ignore_index=True
)

night_transport = gpd.GeoDataFrame(
    night_transport,
    geometry="geometry",
    crs=bus_std.crs
)


# 6. output

night_transport.to_file(
    "night_transport_points.geojson",
    driver="GeoJSON"
)

print("night_transport_points.geojson saved")
print(night_transport.groupby("mode").size())
