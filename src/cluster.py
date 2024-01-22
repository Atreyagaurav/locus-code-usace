import numpy as np
import pandas as pd
import os
import xarray
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from kneed import KneeLocator

from src.huc import HUC
import src.precip as precip


def storm_centers(df: pd.DataFrame):
    m = dimensionality_reduction(df)
    labels = clustering(m)
    df["cluster"] = labels
    return df


def dimensionality_reduction(df: pd.DataFrame):
    pca = PCA(n_components=20)
    df_norm = df.apply(lambda row: row / row.sum(), axis=1)
    return pca.fit_transform(df_norm.to_numpy())


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



def clustered_df(huc: HUC, series, ndays):
    if series == "ams":
        grids = precip.load_ams_grids(huc, ndays)
    elif series == "pds":
        threshold = precip.get_threhold(huc, ndays)
        grids = precip.load_pds_grids(huc, ndays, threshold)
    df_fname = huc.data_path(f"clusters-{series}_{ndays}day.csv")
    try:
        cluster = pd.read_csv(df_fname, index_col="end_date")
        return grids.join(cluster)
    except FileNotFoundError:
        # cluster details
        df_clustered = storm_centers(grids)
        df_clustered.cluster = df_clustered.cluster.map(lambda c: f"C-{c+1}")
        df_clustered.cluster.to_csv(df_fname)
        return df_clustered


def cluster_means(huc: HUC, series, ndays):
    df_clustered = clustered_df(huc, series, ndays)
    means = df_clustered.groupby("cluster").mean().T

    ids = pd.read_csv(huc.data_path("ids.csv"), index_col="ids")
    ids.set_index(pd.Index(ids.index, dtype=int), inplace=True)

    cluster_means = ids.join(
        means.set_index(pd.Index(means.index.map(float), dtype=int))
    )
    weights = huc.weights.to_dataframe().dropna()
    weights.set_index(pd.Index(weights.ids, dtype=int), inplace=True)
    summary = pd.DataFrame(
        {"count": df_clustered.cluster.value_counts()},
        index=sorted(df_clustered.cluster.unique()),
        columns=["precip", "count"])
    for ind in summary.index:
        grid_precip = cluster_means.loc[:, [ind]]
        grid_precip = grid_precip.join(weights.weights)
        prec = grid_precip.weights * grid_precip.loc[:, ind]
        summary.loc[ind, "precip"] = prec.sum()
    summary.index.name = "cluster"
    summary.to_csv(
        huc.data_path(f"clusters-summary-{series}_{ndays}day.csv"))
    return cluster_means


def cluster_weights(huc: HUC, series, ndays):
    filename = huc.data_path(f"clusters-weights_{series}_{ndays}day.nc")
    if os.path.exists(filename):
        return xarray.open_dataset(filename)
    means = cluster_means(huc, series, ndays)
    clusters = [c for c in means.columns if c.startswith("C-")]
    clusters_wt = xarray.Dataset(coords=huc.weights.coords).assign_coords(
        {"cluster": clusters}
    )
    clusters_wt["prec"] = xarray.DataArray(np.nan, coords=clusters_wt.coords)
    ids2latlon = {v:k for k,v in huc.weights.ids.to_series().dropna().to_dict().items()}
    lat_ind = {v:i for i,v in enumerate(clusters_wt.lat.to_numpy())}
    lon_ind = {v:i for i,v in enumerate(clusters_wt.lon.to_numpy())}
    clus_ind = {v:i for i,v in enumerate(clusters_wt.cluster.to_numpy())}
    for c in clusters:
        cluster_slice = means.loc[:, ["lat", "lon", c]].dropna()
        for i,row in cluster_slice.iterrows():
            ind = lat_ind[row.lat],lon_ind[row.lon],clus_ind[c]
            clusters_wt.prec[ind] = row.loc[c]
    clusters_wt["w_prec"] = clusters_wt.prec * huc.weights.weights
    clusters_wt["weights"] = clusters_wt.prec / clusters_wt.prec.sum(dim=["lat", "lon"])
    clusters_wt.to_netcdf(filename)
    return clusters_wt

