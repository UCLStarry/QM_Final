# -*- coding: utf-8 -*-
"""
Created on 2025/12/24 00:41

@author: starry
"""

# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


INPUT_PATH = "outputs/lsoa_night_transport_venues_summary.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

X_COL = "night_transport_density_km2"
IMD_RANK_COL = "IMD_RANK"
GROUP_MODE = "quintile"
EXISTING_GROUP_COL = "cluster"


USE_LOG1P = True
CLIP_Q = 0.995

FIG_PATH = os.path.join(OUT_DIR, f"raincloud_{X_COL}_{GROUP_MODE}.png")



# 1. Load file：csv / geojson / shp

def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in [".geojson", ".json", ".gpkg", ".shp"]:
        import geopandas as gpd
        gdf = gpd.read_file(path)
        if "geometry" in gdf.columns:
            gdf = gdf.drop(columns=["geometry"])
        return pd.DataFrame(gdf)
    raise ValueError(f"Unsupported file type: {ext}")





def half_kde_fill(ax, x, y0, height=0.33, gridsize=256, color=None, alpha=0.65):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return

    kde = gaussian_kde(x)
    xs = np.linspace(np.min(x), np.max(x), gridsize)
    dens = kde(xs)
    dens = dens / dens.max() * height

    ax.fill_between(xs, y0, y0 + dens, color=color, alpha=alpha, linewidth=0)




def rain_jitter(ax, x, y0, color=None, alpha=0.45, s=9, jitter=0.12, y_shift=-0.12, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return

    rng = np.random.default_rng(seed)
    y = y0 + y_shift + rng.uniform(-jitter, jitter, size=len(x))
    ax.scatter(x, y, s=s, color=color, alpha=alpha, edgecolors="none")




def raincloud_like_ggdist(
    df: pd.DataFrame,
    x_col: str,
    group_col: str,
    order=None,
    title=None,
    xlabel=None,
    ylabel=None,
    palette_name="Set2",
    figsize=(11, 6),
    clip_q=0.995,
    savepath=None,
):
    d = df[[x_col, group_col]].dropna()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d = d.dropna(subset=[x_col])
    d[group_col] = d[group_col].astype(str)

    if order is None:
        order = list(pd.unique(d[group_col]))

    # color palette
    cmap = plt.get_cmap(palette_name)
    colors = {g: cmap(i % cmap.N) for i, g in enumerate(order)}

    fig, ax = plt.subplots(figsize=figsize)


    if clip_q is not None:
        xmax = np.nanquantile(d[x_col].values, clip_q)
        d = d[d[x_col] <= xmax]
    else:
        xmax = np.nanmax(d[x_col].values)

    for i, g in enumerate(order):
        x = d.loc[d[group_col] == g, x_col].values
        if len(x) == 0:
            continue
        y0 = i
        c = colors[g]


        half_kde_fill(ax, x, y0=y0, height=0.33, color=c, alpha=0.65)


        rain_jitter(ax, x, y0=y0, color=c, alpha=0.45, s=9, jitter=0.11, y_shift=-0.12, seed=42+i)

        # IQR + median
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        ax.plot([q1, q3], [y0, y0], color="black", linewidth=4, solid_capstyle="butt", zorder=5)
        ax.scatter([med], [y0], color="black", s=70, zorder=6)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_title(title or f"Rain Cloud Plot of {x_col} by {group_col}")
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or group_col)
    ax.grid(True, axis="x", alpha=0.2)
    ax.set_xlim(left=0, right=xmax * 1.02)

    # legend
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10, color=colors[g], label=g)
        for g in order
    ]
    ax.legend(handles=handles, title=group_col, bbox_to_anchor=(1.02, 0.5), loc="center left", frameon=False)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    return fig, ax



# main

def main():
    df = read_any(INPUT_PATH)
    print("[OK] loaded:", INPUT_PATH)
    if X_COL not in df.columns:
        raise KeyError(f"Missing X_COL: {X_COL}")


    df[X_COL] = pd.to_numeric(df[X_COL], errors="coerce")
    if USE_LOG1P:
        df[X_COL] = np.log1p(df[X_COL])
        x_label = f"log(1 + {X_COL})"
    else:
        x_label = X_COL

    # (quintile/decile/cluster/borough)
    if GROUP_MODE in ["quintile", "decile"]:
        if IMD_RANK_COL not in df.columns:
            raise KeyError(f"Missing IMD rank column: {IMD_RANK_COL}")
        df[IMD_RANK_COL] = pd.to_numeric(df[IMD_RANK_COL], errors="coerce")

        q = 5 if GROUP_MODE == "quintile" else 10
        labels = [f"Q{i}" for i in range(1, q + 1)]
        group_col = f"IMD_{GROUP_MODE}"
        df[group_col] = pd.qcut(df[IMD_RANK_COL], q=q, labels=labels)
        order = labels
        title = f"Rain Cloud Plot of {x_label} by IMD {GROUP_MODE.title()}"

    elif GROUP_MODE == "existing":
        if EXISTING_GROUP_COL not in df.columns:
            raise KeyError(f"Missing group col: {EXISTING_GROUP_COL}")
        group_col = EXISTING_GROUP_COL
        order = None
        title = f"Rain Cloud Plot of {x_label} by {group_col}"

    else:
        raise ValueError("GROUP_MODE must be 'quintile', 'decile', or 'existing'.")

    fig, ax = raincloud_like_ggdist(
        df,
        x_col=X_COL,
        group_col=group_col,
        order=order,
        title=title,
        xlabel=x_label,
        ylabel=group_col,
        clip_q=CLIP_Q,
        savepath=FIG_PATH,
    )
    print("[OK] saved:", FIG_PATH)
    plt.show()


if __name__ == "__main__":
    main()
