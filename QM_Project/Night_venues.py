# -*- coding: utf-8 -*-
"""
Created on 2025/12/21 23:19

@author: starry
"""

# -*- coding: utf-8 -*-
"""
OSM night venues (pub/bar/nightclub) -> join to LSOA -> merge into existing LSOA summary
Outputs:
  - outputs/night_venues_points.geojson
  - outputs/lsoa_night_transport_venues_summary.geojson
  - outputs/lsoa_night_transport_venues_summary.csv
"""

import os
import time
import math
import json
import requests
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point


# Paths / config

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

#  Night Transport + IMD
LSOA_SUMMARY_IN = os.path.join(OUT_DIR, "lsoa_night_transport_imd_summary.geojson")

# Output
VENUES_PTS_OUT = os.path.join(OUT_DIR, "night_venues_points.geojson")
LSOA_SUMMARY_OUT_GEOJSON = os.path.join(OUT_DIR, "lsoa_night_transport_venues_summary.geojson")
LSOA_SUMMARY_OUT_CSV = os.path.join(OUT_DIR, "lsoa_night_transport_venues_summary.csv")

# Overpass endpoint (main)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


VENUE_TAGS = [
    ("amenity", "pub"),
    ("amenity", "bar"),
    ("amenity", "nightclub"),
]

# Column names
LSOA_ID_COL_CANDIDATES = ["LSOA11CD", "lsoa11cd", "LSOA21CD", "lsoa21cd"]
TRANSPORT_PTS_COL_CANDIDATES = ["night_transport_pts", "night_transport_points", "transport_pts", "night_pts"]


# Helpers

def pick_existing_col(gdf: gpd.GeoDataFrame, candidates):
    for c in candidates:
        if c in gdf.columns:
            return c
    raise KeyError(f"None of the candidate columns exist: {candidates}\nExisting: {list(gdf.columns)}")

def overpass_query_for_bbox(bbox, tags):
    """
    bbox: (minx, miny, maxx, maxy) in WGS84 lon/lat
    tags: list of (k, v)
    """
    minx, miny, maxx, maxy = bbox
    parts = []
    for k, v in tags:
        # nodes + ways + relations; for ways/relations ask center
        parts.append(f'node["{k}"="{v}"]({miny},{minx},{maxy},{maxx});')
        parts.append(f'way["{k}"="{v}"]({miny},{minx},{maxy},{maxx});')
        parts.append(f'relation["{k}"="{v}"]({miny},{minx},{maxy},{maxx});')

    body = "\n".join(parts)
    query = f"""
    [out:json][timeout:180];
    (
      {body}
    );
    out center tags;
    """
    return query

def fetch_overpass_elements(query, max_retries=4, backoff_sec=8):
    """
    Fetch Overpass with basic retry/backoff (helps with 429 / 504).
    """
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(OVERPASS_URL, data=query.encode("utf-8"), headers={"Content-Type": "application/x-www-form-urlencoded"})
            if r.status_code == 200:
                return r.json().get("elements", [])
            else:
                # retry on typical transient errors
                if r.status_code in (429, 502, 503, 504):
                    time.sleep(backoff_sec * attempt)
                    continue
                raise RuntimeError(f"Overpass HTTP {r.status_code}: {r.text[:2000]}")
        except requests.RequestException as e:
            if attempt == max_retries:
                raise
            time.sleep(backoff_sec * attempt)
    return []

def elements_to_gdf(elements):
    """
    Turn Overpass elements into a point GeoDataFrame.
    nodes have lat/lon; ways/relations may have center.
    """
    rows = []
    for el in elements:
        el_type = el.get("type")
        el_id = el.get("id")
        tags = el.get("tags", {}) or {}

        # coordinates
        if el_type == "node":
            lat = el.get("lat")
            lon = el.get("lon")
        else:
            center = el.get("center", {})
            lat = center.get("lat")
            lon = center.get("lon")

        if lat is None or lon is None:
            continue

        rows.append({
            "osm_type": el_type,
            "osm_id": int(el_id),
            "name": tags.get("name"),
            "amenity": tags.get("amenity"),
            "geometry": Point(float(lon), float(lat)),
            "tags_json": json.dumps(tags, ensure_ascii=False),
        })

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    # Deduplicate: same (osm_type, osm_id) should be unique
    if not gdf.empty:
        gdf = gdf.drop_duplicates(subset=["osm_type", "osm_id"]).reset_index(drop=True)
    return gdf

def ensure_area_km2(lsoa_gdf):
    """
    Ensure there is an area_km2 column (computed from projected CRS).
    """
    if "area_km2" in lsoa_gdf.columns and pd.api.types.is_numeric_dtype(lsoa_gdf["area_km2"]):
        return lsoa_gdf

    # Use British National Grid for accurate area
    lsoa_proj = lsoa_gdf.to_crs("EPSG:27700")
    lsoa_gdf["area_km2"] = (lsoa_proj.geometry.area / 1_000_000).astype(float)
    return lsoa_gdf


# Main

def main():
    if not os.path.exists(LSOA_SUMMARY_IN):
        raise FileNotFoundError(f"Missing input: {LSOA_SUMMARY_IN}")

    lsoa = gpd.read_file(LSOA_SUMMARY_IN)

    lsoa_id_col = pick_existing_col(lsoa, LSOA_ID_COL_CANDIDATES)
    transport_col = pick_existing_col(lsoa, TRANSPORT_PTS_COL_CANDIDATES)

    # Make sure CRS exists
    if lsoa.crs is None:

        lsoa = lsoa.set_crs("EPSG:27700")

    # Prepare bbox in WGS84 for Overpass
    lsoa_wgs84 = lsoa.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = lsoa_wgs84.total_bounds
    bbox = (minx, miny, maxx, maxy)

    query = overpass_query_for_bbox(bbox, VENUE_TAGS)
    elements = fetch_overpass_elements(query)
    venues = elements_to_gdf(elements)

    if venues.empty:
        print("No venues returned from Overpass for this bbox. Check bbox / tags / endpoint.")
        # still write empty outputs for pipeline stability
        venues.to_file(VENUES_PTS_OUT, driver="GeoJSON")
        lsoa.to_file(LSOA_SUMMARY_OUT_GEOJSON, driver="GeoJSON")
        lsoa.drop(columns="geometry").to_csv(LSOA_SUMMARY_OUT_CSV, index=False)
        return

    # Spatial join: venues -> LSOA
    # Use LSOA CRS for join
    venues_in_lsoa_crs = venues.to_crs(lsoa.crs)

    # join predicate: within
    joined = gpd.sjoin(venues_in_lsoa_crs, lsoa[[lsoa_id_col, "geometry"]], how="left", predicate="within")
    venues_in = joined.dropna(subset=[lsoa_id_col]).copy()

    # Aggregate counts per LSOA
    counts = venues_in.groupby(lsoa_id_col).size().rename("night_venues_pts").reset_index()

    # Merge back to LSOA table
    lsoa2 = lsoa.merge(counts, on=lsoa_id_col, how="left")
    lsoa2["night_venues_pts"] = lsoa2["night_venues_pts"].fillna(0).astype(int)

    # area + density
    lsoa2 = ensure_area_km2(lsoa2)
    lsoa2["night_venues_density_km2"] = lsoa2["night_venues_pts"] / lsoa2["area_km2"].replace({0: float("nan")})

    # venues_to_transport_ratio

    denom = pd.to_numeric(lsoa2[transport_col], errors="coerce").fillna(0).astype(float)
    lsoa2["venues_to_transport_ratio"] = lsoa2["night_venues_pts"] / denom.replace({0: float("nan")})

    # Save venues points
    venues_out = venues_in[[c for c in venues_in.columns if c not in ("index_right",)]].copy()
    venues_out = venues_out.to_crs("EPSG:4326")
    venues_out.to_file(VENUES_PTS_OUT, driver="GeoJSON")

    # Save updated LSOA summary
    lsoa2.to_file(LSOA_SUMMARY_OUT_GEOJSON, driver="GeoJSON")
    lsoa2.drop(columns="geometry").to_csv(LSOA_SUMMARY_OUT_CSV, index=False)

    print("Done.")
    print(f"- Venues points: {VENUES_PTS_OUT}")
    print(f"- LSOA summary (geojson): {LSOA_SUMMARY_OUT_GEOJSON}")
    print(f"- LSOA summary (csv): {LSOA_SUMMARY_OUT_CSV}")
    print(f"Used columns: LSOA_ID={lsoa_id_col}, transport_pts={transport_col}")

if __name__ == "__main__":
    main()
