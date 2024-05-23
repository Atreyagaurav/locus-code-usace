import numpy as np
import pandas as pd
import os
import sys
import xarray
import lmfit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_samples, silhouette_score

from src.huc import HUC
import src.precip as precip
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def get_center(huc, dates):
    prec = huc.get_gridded_df(pd.DatetimeIndex(dates))
    prec = prec.set_index("time", append=True).loc[:, ["ids", "prec"]].to_xarray()
    prec = prec.prec.where(prec.ids.notnull())
    centers = pd.DataFrame(columns=["x", "y"], index=prec.time)
    for t in prec.time.values:
        p = prec.sel(time=t).drop_vars("time")
        # grid2 = gaussian_filter(grid, sigma=2, order=0)
        threshold = p.quantile(0.95)
        center = np.where(p > threshold)

        centers.loc[t, "x"] = float(p.lon[center[1]].mean())
        centers.loc[t, "y"] = float(p.lat[center[0]].mean())
    return centers


def storm_centers(df: pd.DataFrame, huc, series, ndays):
    m = None
    labels = clustering(m, huc, series, ndays, use_centers=True)
    df.loc[:, "cluster"] = labels
    return df


def plot_cluster_analysis(row, labels, sample_scores, X, centers, filename):
    n_clusters = row.name
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)
    
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(labels) + (n_clusters + 1) * 10])
    y_lower = 10
    for i in range(1, n_clusters+1):
        ith_cluster_sil_values = sample_scores[labels == i]
        ith_cluster_sil_values.sort()
        size_cluster_i = ith_cluster_sil_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_sil_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
        
    ax1.set_title(f"N = {row.name}, silhoutte={row.silhouette_score:.4f}, sse={row.sse:.4f}")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=sample_scores.mean(), color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % (i+1), alpha=1, s=50, edgecolor="k")

    ax2.set_title(f"converged = {row.converged} counts = {row.counts}")
    ax2.set_xlabel("longitude")
    ax2.set_ylabel("latitude")
    fig.savefig(filename)
    print(filename, file=sys.stderr)


def best_number_of_clusters(m: np.ndarray, huc, series, ndays, centers_latlon):
    summary = pd.DataFrame(
        index=range(2,10),
        columns=["converged", "sse", "silhouette_score", "counts"]
    )
    labels = {}
    for k, row in summary.iterrows():
        kmeans = KMeans(n_clusters=k, n_init=100).fit(m)
        row.sse = kmeans.inertia_
        row.converged = kmeans.n_iter_ < kmeans.max_iter
        row.silhouette_score = silhouette_score(m, kmeans.labels_)
        sample_scores = silhouette_samples(m, kmeans.labels_)
        clusters_map = {
            c: i+1 for i,c in enumerate(
                pd.Series(kmeans.labels_).value_counts().index
            )
        }
        labels[k] = [clusters_map[l] for l in kmeans.labels_]
        row.counts = ' '.join(map(str, pd.Series(labels[k]).value_counts()))
        plot_cluster_analysis(
            row,
            np.array(labels[k]),
            sample_scores,
            centers_latlon.values,
            centers_latlon.groupby(labels[k]).mean().values,
            huc.image_path(f"{series}_{ndays}day_kmeans-{k}N.png")
        )

    knee = KneeLocator(summary.index, summary.sse, curve="convex",
                       direction="decreasing").elbow
    summary.to_csv(huc.data_path(f"clusters-{series}_{ndays}day_summary.csv"))
    return summary.loc[knee, :], labels[knee]


def clustering(m: np.ndarray, huc, series, ndays, use_centers=False):
    if series == "ams":
        dates = precip.load_ams_series(huc, ndays).end_date
    elif series == "pds":
        threshold = precip.get_threhold(huc, ndays)
        dates = precip.load_pds_series(huc, ndays, threshold).end_date
    centers_latlon = get_center(huc, dates)
    if use_centers:
        m = centers_latlon.values
    
    best, labels = best_number_of_clusters(m, huc, series, ndays, centers_latlon)
    sample_scores = silhouette_samples(m, labels)
    means = pd.Series(sample_scores).groupby(labels).mean()
    print(f"{series}_{ndays}day_converged={best.converged}")
    print(f"{series}_{ndays}day_num_cluster={best.name}")
    print(f"{series}_{ndays}day_avg_silhouette={best.silhouette_score}")
    print(f"{series}_{ndays}day_lta_silhouette={(means < best.silhouette_score).sum()}")
    print(f"{series}_{ndays}day_neg_silhouette={(means < 0).sum()}")
    print(f"{series}_{ndays}day_cluster_counts={best.counts}")
    return labels


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
        df_clustered = storm_centers(grids, huc, series, ndays)
        df_clustered.cluster = df_clustered.cluster.map(lambda c: f"C-{c}")
        df_clustered.cluster.to_csv(df_fname)
        return df_clustered


def cluster_means(huc: HUC, series, ndays):
    df_clustered = clustered_df(huc, series, ndays)
    clusters = df_clustered.pop("cluster")
    means_c = df_clustered.groupby(clusters).mean().T

    ids = pd.read_csv(huc.data_path("ids.csv"), index_col="ids")
    ids.set_index(pd.Index(ids.index, dtype=int), inplace=True)

    means = means_c.apply(lambda r: r * ids.weights).apply(lambda row: row / row.mean(), axis=1)
    cluster_means = ids.join(
        means.set_index(pd.Index(means.index.map(float), dtype=int))
    )
    summary = huc.rolling_timeseries(ndays).loc[df_clustered.index, :].groupby(clusters).describe()
    summary.columns = [c for _,c in summary.columns]
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
    clusters_wt["weights"] = xarray.DataArray(np.nan, coords=clusters_wt.coords)
    ids2latlon = {v: k for k, v in huc.weights.ids.to_series().dropna().to_dict().items()}
    lat_ind = {v: i for i, v in enumerate(clusters_wt.lat.to_numpy())}
    lon_ind = {v: i for i, v in enumerate(clusters_wt.lon.to_numpy())}
    clus_ind = {v: i for i, v in enumerate(clusters_wt.cluster.to_numpy())}
    for c in clusters:
        cluster_slice = means.loc[:, ["lat", "lon", c]].dropna()
        for i, row in cluster_slice.iterrows():
            ind = lat_ind[row.lat], lon_ind[row.lon], clus_ind[c]
            clusters_wt.weights[ind] = row.loc[c]
    clusters_wt.to_netcdf(filename)
    return clusters_wt
