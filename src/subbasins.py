from typing import Tuple
import xarray
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from src.huc import HUC
from src.cluster import cluster_weights


def get_subbasins(huc: HUC) -> Tuple[str, str]:
    N = len(huc.huc_code)
    subs = {code:HUC(code) for code,_ in
            filter(lambda h: h[0].startswith(huc.huc_code),
                   HUC.all_huc_codes(N+2))}
    return subs


def _subbasin_dataset(huc: HUC):
    sub_basins = get_subbasins(huc)
    subs = list(sub_basins.keys())
    huc.load_weights(calculate=True)
    weights = xarray.Dataset().assign_coords({
        "lat": huc.weights.lat,
        "lon": huc.weights.lon,
        "subbasin": subs
    })
    zeros = np.zeros((len(weights.lat), len(weights.lon), len(weights.subbasin)))
    weights["weights"] = xarray.DataArray(zeros, dims=["lat", "lon", "subbasin"])
    for sub in subs:
        sub_basins[sub].load_weights(calculate=True)
        new_val = sub_basins[sub].weights.weights
        weights.weights.loc[{"lat":new_val.lat, "lon":new_val.lon, "subbasin":sub}] = new_val 
        weights["weights"] = weights.weights / weights.weights.sum(dim=["subbasin"])
        weights.to_netcdf(huc.data_path("subbasins-weights.nc"))
    return weights


def subbasin_dataset(huc: HUC):
    try:
        return xarray.load_dataset(huc.data_path("subbasins-weights.nc"))
    except FileNotFoundError:
        return _subbasin_dataset(huc)

    
def _subbasins_weights(huc: HUC, series: str, ndays: int) -> pd.DataFrame:
    clusters = cluster_weights(huc, series, ndays)
    subweights = subbasin_dataset(huc)
    sub_div = subweights * clusters.weights
    df = sub_div.sum(dim=["lat", "lon"]).to_dataframe().unstack('cluster')
    df.columns = [c for _,c in df.columns]
    df.to_csv(huc.data_path(f"subbasins-weights-{series}-{ndays}day.csv"))
    return df


def subbasins_weights(huc: HUC, series: str, ndays: int) -> pd.DataFrame:
    try:
        return pd.read_csv(huc.data_path(f"subbasins-weights-{series}-{ndays}day.csv"), index_col="subbasin", dtype={"subbasin": str})
    except FileNotFoundError:
        return _subbasins_weights(huc, series, ndays)


def subbasins_gdf(huc:HUC, series: str, ndays: int):
    sub_basins = get_subbasins(huc)
    gdf = pd.concat([h.geometry_as_geodataframe() for h in sub_basins.values()])
    weights = subbasins_weights(huc, series, ndays)
    gdf = gdf.set_index(f"huc{len(huc.huc_code)+2}").join(weights)
    return gdf
