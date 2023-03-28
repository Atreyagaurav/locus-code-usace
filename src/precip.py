from typing import List

import pandas as pd
from huc import HUC
from datetime import timedelta


def process_huc_basin(huc_code: str, years: List[int], ndays: int):
    huc = HUC(huc_code)
    ams = calculate_ams_series(huc, years, ndays)
    ams.to_csv(huc.data_path(f"ams_{ndays}dy_series.csv"))
    ams_grids = calculate_series_grids(huc, ams)
    ams_grids.to_csv(huc.data_path(f"ams_{ndays}dy_grids.csv"))
    pds = calculate_pds_series(huc, years, ndays, ams.p_mm.min())
    pds.to_csv(huc.data_path(f"pds_{ndays}dy_series.csv"))
    pds_grids = calculate_series_grids(huc, pds)
    pds_grids.to_csv(huc.data_path(f"pds_{ndays}dy_grids.csv"))


def clustering():
    # just copy pasted what I did to get the graph, need to control it further.
    df_clustered = cluster.storm_centers(ams_grids)
    ids = pd.read_csv(huc.data_path("ids.csv"), index_col="ids")
    ids = gpd.GeoDataFrame(ids, geometry=gpd.points_from_xy(ids['lat'], ids['lon']))
    cluster_means = ids.join(df_clustered.groupby("cluster").mean().T)
    nclusters = len(df_clustered.cluster.unique())
    fig, axs = plt.subplots(nrows=nclusters,
                            ncols=2,
                            figsize=(20, 30),
                            sharex=True, sharey=True)
    for i in range(nclusters):
        cluster_means.plot(ax=axs[i, 0],
                           column=i,
                           vmin=20,
                           vmax=110,
                           legend=True) 
        cluster_means.plot(ax=axs[i, 1],
                           column=i,
                           legend=True)
        axs[i, 0].set_title(f'cluster: {i}')
        axs[i, 1].set_title(f'cluster: {i}')
    plt.savefig(f"./images/{huc.huc_code}/ams_1dy.png")



def calculate_ams_series(huc: HUC, years: List[int], ndays: int) -> pd.DataFrame:
    basin = huc.rolling_timeseries(years, ndays)
    year = pd.DatetimeIndex(basin.index).year
    series = basin[basin == basin.groupby(year).transform(max)].dropna()
    ams_series = pd.DataFrame({
        "p_mm": series.prec,
        "end_date": series.index
    }).reset_index().drop(columns=["time"])
    ams_series.loc[:, 'duration'] = ndays
    return ams_series


def calculate_pds_series(huc: HUC,
                         years: List[int],
                         ndays: int,
                         threshold: float) -> pd.DataFrame:
    basin = huc.rolling_timeseries(years, ndays)
    series = basin[basin.prec > threshold].dropna()
    pds_series = pd.DataFrame({
        "p_mm": series.prec,
        "end_date": series.index
    }).reset_index().drop(columns=["time"])
    pds_series.loc[:, 'duration'] = ndays
    return pds_series


def calculate_series_grids(huc: HUC, ams: pd.DataFrame) -> pd.DataFrame:
    def grids():
        for ind, row in ams.iterrows():
            end = pd.to_datetime(row.end_date)
            dates = [(end - timedelta(days=i)) for i in range(row.duration)]
            yield huc.get_gridded_df(dates).groupby("ids").prec.sum().to_dict()
    all_grids = pd.DataFrame(list(grids()), index=ams.end_date, dtype=float)
    return all_grids
