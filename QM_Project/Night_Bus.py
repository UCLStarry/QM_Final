# -*- coding: utf-8 -*-
"""
Created on 2025/12/21 21:40

@author: starry
"""

import re
import time
import requests
import pandas as pd
import geopandas as gpd

BASE = "https://api.tfl.gov.uk"
APP_KEY = "fdc6b77d604a46a2975a18379b9493c1"

def tfl_get(path, params=None, retries=3, sleep=0.2):
    if params is None:
        params = {}
    # attach credentials
    params.update({"app_key": APP_KEY})
    url = f"{BASE}{path}"
    for i in range(retries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        time.sleep(1 + i)
    raise RuntimeError(f"TfL request failed: {r.status_code} {r.text[:200]}")

def get_all_bus_lines():
    # List lines by mode = bus
    # Swagger: /Line/Mode/{modes}
    lines = tfl_get("/Line/Mode/bus")
    df = pd.DataFrame(lines)
    return df

def filter_night_bus_lines(df_lines):
    # Most London night bus routes are prefixed with "N"
    df = df_lines.copy()
    df["is_night_bus"] = df["name"].astype(str).str.match(r"^N\d+", na=False)
    return df[df["is_night_bus"]].reset_index(drop=True)

def extract_stops_from_route_sequence(route_seq_json):
    # route_seq_json contains stopPointSequences -> stopPoint[] with lat/lon
    stops = []
    for sps in route_seq_json.get("stopPointSequences", []):
        for sp in sps.get("stopPoint", []):
            # fields usually include id/name/lat/lon; fallbacks included
            stops.append({
                "stop_id": sp.get("id"),
                "stop_name": sp.get("name"),
                "lat": sp.get("lat"),
                "lon": sp.get("lon"),
            })
    return stops

def get_line_stops(line_id):
    all_stops = []
    for direction in ["inbound", "outbound"]:
        js = tfl_get(f"/Line/{line_id}/Route/Sequence/{direction}", params={"serviceTypes": "Regular"})
        all_stops.extend(extract_stops_from_route_sequence(js))
    return all_stops

def build_night_bus_stop_layer(out_geojson="night_bus_stops.geojson", out_lines_csv="night_bus_lines.csv"):
    lines_df = get_all_bus_lines()
    night_lines = filter_night_bus_lines(lines_df)

    # Save line list
    night_lines[["id", "name", "modeName"]].to_csv(out_lines_csv, index=False, encoding="utf-8")
    print(f"Saved night bus line list -> {out_lines_csv} ({len(night_lines)} lines)")

    # Collect stops
    stop_rows = []
    for idx, row in night_lines.iterrows():
        line_id = row["id"]
        line_name = row["name"]
        try:
            stops = get_line_stops(line_id)
            for s in stops:
                s["line_id"] = line_id
                s["line_name"] = line_name
                stop_rows.append(s)
        except Exception as e:
            print(f"[WARN] line {line_name} ({line_id}) failed: {e}")

        time.sleep(0.15)

    df = pd.DataFrame(stop_rows).dropna(subset=["lat", "lon", "stop_id"])
    # de-duplicate by stop_id (keep one point)
    df_unique = df.sort_values(["stop_id", "line_name"]).drop_duplicates("stop_id", keep="first").reset_index(drop=True)

    gdf = gpd.GeoDataFrame(
        df_unique,
        geometry=gpd.points_from_xy(df_unique["lon"], df_unique["lat"]),
        crs="EPSG:4326"
    )
    gdf.to_file(out_geojson, driver="GeoJSON")
    print(f"Saved night bus stops -> {out_geojson} ({len(gdf)} unique stops)")

if __name__ == "__main__":
    build_night_bus_stop_layer()
