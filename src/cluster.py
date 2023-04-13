import numpy as np
import pandas as pd
import geopandas as gpd
import os
import shapely
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from kneed import KneeLocator

from src.livneh import LivnehData
from src.huc import HUC
import src.precip as precip


def storm_centers(df: pd.DataFrame):
    m = dimensionality_reduction(df)
    labels = clustering(m)
    df["cluster"] = labels
    return df


def dimensionality_reduction(df: pd.DataFrame):
    pca = PCA(n_components=20)
    return pca.fit_transform(StandardScaler().fit_transform(df.to_numpy()))


def clustering(m: np.ndarray):
    sse = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k).fit(m)  # m[:,:4]
        sse.append(kmeans.inertia_)
    knee = KneeLocator(range(2, 11), sse, curve="convex",
                       direction="decreasing").elbow
    clusters = KMeans(n_clusters=knee).fit(m)  # m[:,:4]
    return clusters.labels_


def normalize_cluster(df: pd.DataFrame, ids: pd.DataFrame):
    nclusters = sum([1 if type(c) is int else 0 for c in df.columns.values])
    df.sort_values(by="id", inplace=True)
    ids.sort_values(by="id", inplace=True)
    for i in range(nclusters):
        basin = sum(
            [p * w for (p, w) in zip(df[i].to_numpy(),
                                     ids["area_weight"].to_numpy())]
        )
        df[f"{i}norm"] = df[i].to_numpy() / basin
    return df


def cluster_means(huc: HUC, series, ndays):
    filename = huc.data_path(f"clusters-means_{series}_{ndays}day.shp")
    if os.path.exists(filename):
        return gpd.read_file(filename)

    if series == "ams":
        grids = precip.load_ams_grids(huc, LivnehData.YEARS, ndays)
    elif series == "pds":
        threshold = precip.get_threhold(huc, ndays)
        grids = precip.load_pds_grids(huc, LivnehData.YEARS, ndays, threshold)

    ids = pd.read_csv(huc.data_path("ids.csv"), index_col="ids")
    shift = LivnehData.RESOLUTION / 2
    geometry = [
        shapely.box(lon - shift, lat - shift, lon + shift, lat + shift)
        for lat, lon in zip(ids["lat"], ids["lon"])
    ]
    ids = gpd.GeoDataFrame(ids, geometry=geometry)
    ids.set_index(pd.Index(ids.index, dtype=int), inplace=True)

    # cluster details
    df_clustered = storm_centers(grids)
    df_clustered.cluster = df_clustered.cluster.map(lambda c: f"C-{c+1}")
    df_clustered.cluster.to_csv(
        huc.data_path(f"clusters-{series}_{ndays}day.csv"))
    means = df_clustered.groupby("cluster").mean().T
    cluster_means = ids.join(
        means.set_index(pd.Index(means.index.map(float), dtype=int))
    )
    cluster_means.to_file(filename)
    weights = huc.weights.to_dataframe().dropna()
    weights.set_index(pd.Index(weights.ids, dtype=int), inplace=True)
    summary = pd.DataFrame(
        {"count": df_clustered.cluster.value_counts()},
        index=sorted(df_clustered.cluster.unique()),
        columns=["precip", "count"])
    for ind in summary.index:
        grid_precip = cluster_means.loc[:,[ind]]
        grid_precip = grid_precip.join(weights.weights)
        prec = grid_precip.weights * grid_precip.loc[:, ind]
        summary.loc[ind, "precip"] = prec.sum()
    summary.index.name = "cluster"
    summary.to_csv(
        huc.data_path(f"clusters-summary-{series}_{ndays}day.csv"))
    return cluster_means
