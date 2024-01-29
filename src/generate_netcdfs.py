from src.huc import HUC
import src.cluster as cluster
import pandas as pd
import xarray


def generate_cluster_netcdfs(h: HUC, series="ams", ndays=1):
    wt = cluster.cluster_weights(h, series, ndays)
    # to remove the "actual_range" attr
    wt["lat"] = 0 + wt.lat
    if wt.lon.min() > 180:
        # GIS compatible range
        wt["lon"] = wt.lon - 360
    for cl in range(len(wt.cluster)):
        c1 = wt.weights[:, :, cl].drop_vars("cluster")
        c1 = (c1 * c1.count() * 100).fillna(0).to_dataset()
        c1.to_netcdf(f"exported-dss/{h.name.replace(' ', '')}_{series}{ndays}d_cluster-{cl+1}_prec-100mm.nc")


def generate_mean_netcdf(h: HUC, series="ams", ndays=1):
    wt = cluster.cluster_weights(h, series, ndays)
    wt["lat"] = 0 + wt.lat
    if wt.lon.min() > 180:
        # GIS compatible range
        wt["lon"] = wt.lon - 360
    csum = pd.read_csv(
        h.data_path(f"clusters-summary-{series}_{ndays}day.csv"), index_col="cluster"
    ).loc[:, "count"]
    cweights = xarray.DataArray.from_series(csum / csum.sum())
    average = (wt.weights * cweights).sum(["cluster"])
    wt["lat"] = 0 + wt.lat
    average_ds = (
        average * average.where(average > 0).count() * 100
    ).to_dataset(name="weights")
    average_ds.to_netcdf(f"exported-dss/{h.name.replace(' ', '')}_{series}{ndays}d_average_prec-100mm.nc")


def generate_uniform_netcdf(h: HUC):
    wt = h.weights.weights * 100
    wt["lat"] = 0 + wt.lat
    if wt.lon.min() > 180:
        # GIS compatible range
        wt["lon"] = wt.lon - 360
    wt.to_netcdf(f"exported-dss/{h.name.replace(' ', '')}_uniform_prec-100mm.nc")
