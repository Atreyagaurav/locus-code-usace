import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.ndimage import generate_binary_structure, binary_erosion
import matplotlib.pyplot as plt
import lmfit

from datetime import datetime as dt
from datetime import timedelta as td

from src.huc import HUC


def get_center(h: HUC, date):
    df = h.get_gridded_df([date])
    df.loc[:, "basin"] = 1
    df.loc[df.ids.isna(), "basin"] = 0
    lat_lon = (len(df.index.levels[0]), len(df.index.levels[1]))
    grid = df.prec.to_numpy().reshape(lat_lon)
    basin = df.basin.to_numpy().reshape(lat_lon)
    grid2 = gaussian_filter(grid, sigma=2, order=0)
    masked = np.multiply(grid2, basin)
    center = np.where(masked == masked.max())
    x = df.index.levels[1][center[1][0]]
    y = df.index.levels[0][center[0][0]]

    model = lmfit.models.Gaussian2dModel()
    all_x = [x for _,x in df.index]
    all_y = [y for y,_ in df.index]
    all_z = masked.flatten()
    params = model.guess(all_z, all_x, all_y)
    return (params.get("centerx").value, params.get("centery").value)


def get_vectors(h: HUC, date):
    (x1, y1) = get_center(h, date - td(days=1))
    (x2, y2) = get_center(h, date)
    (x3, y3) = get_center(h, date + td(days=1))
    ux1 = x2 - x1
    uy1 = y2 - y1
    ux2 = x3 - x2
    uy2 = y3 - y2
    return [x1, y1, x2, y2, ux1, uy1, ux2, uy2]


def track_movements(h: HUC, series="ams", ndays=1):
    ams = pd.read_csv(
        h.data_path(f"{series}_{ndays}dy_series.csv"), index_col="end_date"
    ).join(
        pd.read_csv(h.data_path(f"clusters-{series}_{ndays}day.csv"), index_col="end_date")
    )
    vectors = ams.index.map(dt.fromisoformat).map(lambda d: get_vectors(h, d))
    for i, d in enumerate(ams.index):
        ams.loc[d, ["x1", "y1", "x2", "y2", "ux1", "uy1", "ux2", "uy2"]] = vectors[i]
    ams.cluster = ams.cluster.map(lambda c: c.split("-")[1])
    ams.to_csv(h.data_path(f"{series}_{ndays}dy_vectors.csv"))
    means = ams.groupby("cluster").mean()
    means.loc[:, "x1"] = means.loc[:, "x2"] - means.loc[:, "ux1"]
    means.loc[:, "y1"] = means.loc[:, "y2"] - means.loc[:, "uy1"]
    means.to_csv(h.data_path(f"{series}_{ndays}dy_vectors_mean.csv"))
    bbox = h.geometry.bounds
    t = min(abs(bbox[0] - bbox[2]) ,abs(bbox[1] - bbox[3])) / 100
    poly = h.geometry.simplify(t)
    with open(h.data_path("basin.csv"), "w") as w:
        for (x, y) in poly.boundary.coords:
            w.write(f"{x},{y}\n")


# track_movements(HUC("05"))
track_movements(HUC("1203"), ndays=2)
track_movements(HUC("02070002"))

track_movements(HUC("1203"), "pds", ndays=2)
track_movements(HUC("02070002"), "pds")
