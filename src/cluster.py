import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from kneed import KneeLocator


def storm_centers(df: pd.DataFrame):
    m = dimensionality_reduction(df)
    labels = clustering(m)
    df['cluster'] = labels
    return df


def dimensionality_reduction(df: pd.DataFrame):
    pca = PCA(n_components=20)
    return pca.fit_transform(StandardScaler().fit_transform(df.to_numpy()))


def clustering(m: np.ndarray):
    sse = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k).fit(m) #m[:,:4]
        sse.append(kmeans.inertia_)
    knee = KneeLocator(range(2, 11), sse, curve='convex', direction='decreasing').elbow
    clusters = KMeans(n_clusters=knee).fit(m) #m[:,:4]
    return clusters.labels_


def cluster_means(dflabeled: pd.DataFrame, dfgrids: pd.DataFrame):
    cluster_means = dfgrids.copy(deep=True)
    n_clusters = dflabeled.cluster.max()
    for i in range(0, n_clusters + 1):      # loop over clusters
        cluster = dflabeled[dflabeled.cluster == i]
        means = np.zeros(len(dflabeled.columns) - 1) # number of gridcells
        for j in range(0, len(means)):  # loop over grid cells
            means[j] = cluster[str(j)].mean()
        cluster_means[i] = means
    return cluster_means


def normalize_cluster(df: pd.DataFrame, ids: pd.DataFrame):
    nclusters = sum([1 if type(c) is int else 0 for c in df.columns.values])
    df.sort_values(by='id', inplace=True)
    ids.sort_values(by='id', inplace=True)
    for i in range(nclusters):
        basin = sum([p*w for (p, w) in zip(df[i].to_numpy(), ids['area_weight'].to_numpy())])
        df[f'{i}norm'] = df[i].to_numpy() / basin
    return df
