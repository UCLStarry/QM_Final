# -*- coding: utf-8 -*-
"""
Night Tube stations (GeoJSON) from TfL Unified API
Created on 2025/12/21
@author: starry
"""

import time
import requests
import pandas as pd
import geopandas as gpd

BASE = "https://api.tfl.gov.uk"
APP_KEY = "fdc6b77d604a46a2975a18379b9493c1"

# Night Tube lines
NIGHT_TUBE_LINE_IDS = ["central", "jubilee", "northern", "piccadilly", "victoria"]

def tfl_get(path, params=None, retries=3, sleep=0.4):
    if params is None:
        params = {}
    params = dict(params)
    params.update({"app_key": APP_KEY})
    url = f"{BASE}{path}"

    last = None
    for i in range(retries):
        r = requests.get(url, params=params, timeout=60)
        last = r
        if r.status_code == 200:
            return r.json()

        time.sleep(sleep * (2 ** i))
    raise RuntimeError(f"TfL request failed: {last.status_code} {last.text[:300]}")

def get_line_stop_points(line_id: str):
    """
    TfL: /Line/{id}/StopPoints
    returns list of StopPoint objects with lat/lon
    """
    js = tfl_get(f"/Line/{line_id}/StopPoints")
    rows = []
    for sp in js:
        lat = sp.get("lat")
        lon = sp.get("lon")
        if lat is None or lon is None:
            continue
        rows.append({
            "station_id": sp.get("id"),
            "station_name": sp.get("commonName") or sp.get("name"),
            "lat": lat,
            "lon": lon,
            "line_id": line_id
        })
    return rows

def build_night_tube_station_layer(
    out_geojson="night_tube_stations.geojson",
    out_lines_csv="night_tube_lines.csv"
):
    # lines csv
    pd.DataFrame({"id": NIGHT_TUBE_LINE_IDS}).to_csv(out_lines_csv, index=False, encoding="utf-8")
    print(f"Saved night tube line list -> {out_lines_csv}")

    rows = []
    for line_id in NIGHT_TUBE_LINE_IDS:
        try:
            line_rows = get_line_stop_points(line_id)
            rows.extend(line_rows)
            print(f"{line_id}: {len(line_rows)} stop points")
        except Exception as e:
            print(f"[WARN] line {line_id} failed: {e}")
        time.sleep(0.2)

    df = pd.DataFrame(rows).dropna(subset=["station_id", "lat", "lon"])
    # The same station may belong to multiple lines: First, remove duplicates and retain one point.
    df_unique = df.sort_values(["station_id", "line_id"]).drop_duplicates("station_id", keep="first").reset_index(drop=True)

    gdf = gpd.GeoDataFrame(
        df_unique,
        geometry=gpd.points_from_xy(df_unique["lon"], df_unique["lat"]),
        crs="EPSG:4326"
    )
    gdf.to_file(out_geojson, driver="GeoJSON")
    print(f"Saved night tube stations -> {out_geojson} ({len(gdf)} unique stations)")

if __name__ == "__main__":
    build_night_tube_station_layer()
