# -*- coding: utf-8 -*-
"""
Created on 2025/12/21 22:49

@author: starry
"""

# -*- coding: utf-8 -*-
"""
Join night transport points (night bus + night tube) to LSOA + IMD.

Inputs (expected in current folder):
  - night_bus_stops.geojson
  - night_tube_stations.geojson
  - LSOA.shp (and .dbf/.shx/.prj in same folder)
  - IMD.csv

Outputs:
  - outputs/night_transport_points.geojson
  - outputs/night_transport_points_joined.geojson
  - outputs/lsoa_night_transport_imd_summary.geojson
  - outputs/lsoa_night_transport_imd_summary.csv
"""

import os
import warnings
import pandas as pd
import geopandas as gpd

# silence urllib3 LibreSSL warning
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL*")

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

BUS_PATH = "night_bus_stops.geojson"
TUBE_PATH = "night_tube_stations.geojson"


LSOA_SHP_PATH = "LSOA.shp"
IMD_CSV_PATH = "IMD_2019.csv"


def harmonize_points(bus_gdf: gpd.GeoDataFrame, tube_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Make a common schema:
      - point_id
      - point_name
      - mode              (night_bus / night_tube)
      - line_id
      - line_name
      - source_file
      - geometry
    """
    # bus
    b = bus_gdf.copy()
    rename_bus = {
        "stop_id": "point_id",
        "stop_name": "point_name",
        "line_id": "line_id",
        "line_name": "line_name",
    }
    for k, v in rename_bus.items():
        if k in b.columns:
            b = b.rename(columns={k: v})
    b["mode"] = "night_bus"
    b["source_file"] = os.path.basename(BUS_PATH)

    # one row per stop
    if "point_id" in b.columns and "line_name" in b.columns:
        b = (
            b.groupby("point_id", as_index=False)
             .agg({
                 "point_name": "first",
                 "mode": "first",
                 "source_file": "first",
                 "line_name": lambda x: "; ".join(sorted(set([str(i) for i in x if pd.notna(i)]))),
                 "line_id": lambda x: "; ".join(sorted(set([str(i) for i in x if pd.notna(i)]))),
                 "geometry": "first",
             })
        )
        b = gpd.GeoDataFrame(b, geometry="geometry", crs=bus_gdf.crs)

    # tube
    t = tube_gdf.copy()
    rename_tube = {
        "station_id": "point_id",
        "station_name": "point_name",
        "line_id": "line_id",
        "line_name": "line_name",
    }
    for k, v in rename_tube.items():
        if k in t.columns:
            t = t.rename(columns={k: v})
    t["mode"] = "night_tube"
    t["source_file"] = os.path.basename(TUBE_PATH)

    if "point_id" in t.columns:
        t = t.drop_duplicates(subset=["point_id"]).copy()

    # Ensure minimal columns exist
    for df in (b, t):
        for col in ["point_id", "point_name", "line_id", "line_name", "mode", "source_file"]:
            if col not in df.columns:
                df[col] = pd.NA

    keep_cols = ["point_id", "point_name", "mode", "line_id", "line_name", "source_file", "geometry"]
    out = pd.concat([b[keep_cols], t[keep_cols]], ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=bus_gdf.crs or tube_gdf.crs or "EPSG:4326")

    out = out.dropna(subset=["geometry"]).copy()
    out = out.drop_duplicates(subset=["mode", "point_id"]).copy()
    return out


def main():
    # 0) read points
    for p in [BUS_PATH, TUBE_PATH, LSOA_SHP_PATH, IMD_CSV_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing local file: {p}")

    bus = gpd.read_file(BUS_PATH)
    tube = gpd.read_file(TUBE_PATH)

    if bus.crs is None:
        bus.set_crs("EPSG:4326", inplace=True)
    if tube.crs is None:
        tube.set_crs("EPSG:4326", inplace=True)

    night_points = harmonize_points(bus, tube)
    night_points_out = os.path.join(OUT_DIR, "night_transport_points.geojson")
    night_points.to_file(night_points_out, driver="GeoJSON")
    print(f"[OK] saved: {night_points_out} (n={len(night_points)})")

    # 1. read local LSOA + IMD
    lsoa = gpd.read_file(LSOA_SHP_PATH)
    imd = pd.read_csv(IMD_CSV_PATH)

    # if LSOA has no CRS, assume BNG (common for ONS boundaries)
    if lsoa.crs is None:
        lsoa.set_crs("EPSG:27700", inplace=True)

    # 2. light filter polygons to London-ish extent
    lsoa_wgs84 = lsoa.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = night_points.to_crs("EPSG:4326").total_bounds
    pad = 0.35
    lsoa_wgs84 = lsoa_wgs84.cx[minx - pad:maxx + pad, miny - pad:maxy + pad].copy()
    lsoa = lsoa_wgs84.to_crs("EPSG:27700")  # keep in BNG for join/area

    # 3. find LSOA code column
    possible_code_cols = ["LSOA11CD", "lsoa11cd", "LSOA_CODE", "LSOA_code", "geo_code"]
    code_col = None
    for c in possible_code_cols:
        if c in lsoa.columns:
            code_col = c
            break
    if code_col is None:
        for c in lsoa.columns:
            if lsoa[c].astype(str).str.match(r"^E01\d{7}$").any():
                code_col = c
                break
    if code_col is None:
        raise RuntimeError("Cannot find LSOA code column in local LSOA shapefile.")
    lsoa = lsoa.rename(columns={code_col: "LSOA11CD"})

    # 4. IMD code column
    imd_code_col = None
    for c in ["LSOA11CD", "lsoa11cd", "LSOA code (2011)", "lsoa_code", "LSOA_CODE"]:
        if c in imd.columns:
            imd_code_col = c
            break
    if imd_code_col is None:
        for c in imd.columns:
            if imd[c].astype(str).str.match(r"^E01\d{7}$").any():
                imd_code_col = c
                break
    if imd_code_col is None:
        raise RuntimeError("Cannot find LSOA code column in local IMD CSV.")
    imd = imd.rename(columns={imd_code_col: "LSOA11CD"})

    # 4.1 Standardize the column names of IMD to the standard field names
    imd_colmap = {
        "Index of Multiple Deprivation (IMD) Rank": "IMD_RANK",
        "Index of Multiple Deprivation (IMD) Decile": "IMD_DECILE",
        "Index of Multiple Deprivation (IMD) Score": "IMD_SCORE",
        "Local Authority District code (2019)": "LAD19CD",
        "Local Authority District name (2019)": "LAD19NM",
        "LSOA name (2011)": "LSOA11NM",
    }
    for k, v in imd_colmap.items():
        if k in imd.columns:
            imd = imd.rename(columns={k: v})

    # 4.2 Keep only the key columns
    keep = [c for c in ["LSOA11CD", "LSOA11NM", "LAD19CD", "LAD19NM", "IMD_SCORE", "IMD_RANK", "IMD_DECILE"] if c in imd.columns]
    imd = imd[keep].copy()

    # 4.3 clean data
    for c in ["IMD_SCORE", "IMD_RANK", "IMD_DECILE"]:
        if c in imd.columns:
            imd[c] = (
                imd[c].astype(str)
                      .str.replace(",", "", regex=False)
                      .str.strip()
            )
            imd[c] = pd.to_numeric(imd[c], errors="coerce")


    imd["LSOA11CD"] = imd["LSOA11CD"].astype(str).str.strip()


    # 4. reproject points to BNG
    night_points_bng = night_points.to_crs("EPSG:27700")

    # 5. spatial join: points -> LSOA
    joined = gpd.sjoin(night_points_bng, lsoa[["LSOA11CD", "geometry"]], how="left", predicate="within")
    joined = joined.drop(columns=[c for c in joined.columns if c.startswith("index_")], errors="ignore")

    # 6. attribute join: add IMD
    joined = joined.merge(imd, on="LSOA11CD", how="left")

    joined_out = os.path.join(OUT_DIR, "night_transport_points_joined.geojson")
    joined.to_crs("EPSG:4326").to_file(joined_out, driver="GeoJSON")
    print(f"[OK] saved: {joined_out} (n={len(joined)})")

    # 7. aggregate to LSOA
    lsoa2 = lsoa.copy()
    lsoa2["area_km2"] = lsoa2.geometry.area / 1e6

    counts_total = joined.groupby("LSOA11CD").size().rename("night_transport_pts").reset_index()
    counts_mode = (
        joined.groupby(["LSOA11CD", "mode"]).size().rename("n").reset_index()
              .pivot(index="LSOA11CD", columns="mode", values="n")
              .fillna(0)
              .reset_index()
    )
    counts_mode.columns.name = None

    lsoa_sum = lsoa2.merge(counts_total, on="LSOA11CD", how="left")
    lsoa_sum = lsoa_sum.merge(counts_mode, on="LSOA11CD", how="left")

    for c in ["night_transport_pts", "night_bus", "night_tube"]:
        if c in lsoa_sum.columns:
            lsoa_sum[c] = lsoa_sum[c].fillna(0).astype(int)

    lsoa_sum["night_transport_density_km2"] = (
        lsoa_sum["night_transport_pts"] / lsoa_sum["area_km2"].replace({0: pd.NA})
    )

    lsoa_sum = lsoa_sum.merge(imd, on="LSOA11CD", how="left")

    lsoa_geojson_out = os.path.join(OUT_DIR, "lsoa_night_transport_imd_summary.geojson")
    lsoa_csv_out = os.path.join(OUT_DIR, "lsoa_night_transport_imd_summary.csv")

    lsoa_sum.to_crs("EPSG:4326").to_file(lsoa_geojson_out, driver="GeoJSON")
    lsoa_sum.drop(columns="geometry").to_csv(lsoa_csv_out, index=False, encoding="utf-8")

    print(f"[OK] saved: {lsoa_geojson_out}")
    print(f"[OK] saved: {lsoa_csv_out}")


if __name__ == "__main__":
    main()
