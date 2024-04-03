from src.huc import HUC
import src.cluster as cluster
import pandas as pd
import xarray


def set_attrs(obj, **kwargs):
    for k, v in kwargs.items():
        obj[k] = v

# https://docs.unidata.ucar.edu/nug/current/best_practices.html
# http://cfconventions.org/
# http://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html
def prepare_metadata(dat: xarray.Dataset, name, **kwargs):
    # long name for variables
    # order: "date or time" (T), "height or depth" (Z), "latitude" (Y), or "longitude" (X)
    set_attrs(
        dat.attrs,
        Conventions = "CF-1.11",
        # name will have HUC and cluster-N / mean / uniform
        name = name,
        title = f"Locus: Spatial Distribution of Extreme Precipitation",
        projection = "Geographic",
        institution = "University of Cincinnati ...",
        source = "locus-v0.1",
        history = "Created by clustering the data from Livneh et al., 2013 for each HUC basins after filtering only the extreme precipitations",
        references = "TODO (unpublished)",
        comment = "TODO (Link to github repo, probably)",
    )
    set_attrs(
        dat.coords['lat'].attrs,
        standard_name = "grid_latitude",
        units = "degrees_north",
        axis = "Y"
    )
    set_attrs(
        dat.coords['lon'].attrs,
        standard_name = "grid_longitude",
        units = "degrees_east",
        axis = "X"
    )
    # unit for precipitation_flux is kg m^-2 so it's not true
    # dat.variables['weights'].attrs["standard_name"] = "precipitation_flux"
    set_attrs(
        dat.variables['weights'].attrs,
        long_name = "precipitation weights",
        description = (
            "amount of precipitation that'd fall at the particular" +
            " grid out of total 100 mm precipitation throughout the basin"
        ),
        units = "mm/day",
    )

    try:
        # uniform one doesn't have these variables
        set_attrs(
            dat.variables['date'].attrs,
	    standard_name = "time",
	    calendar = "standard",
	    coordinate_defines = "start",
	    long_name = "Time",
	    delta_t = "0000-00-01 00:00:00",
	    units = "days since 1915-1-1 00:00:0.0",
	    axis = "T",
        )
        set_attrs(
            dat.variables['prec'].attrs,
            long_name = "precipitation",
            description = "basin averaged precipitation depth at that day",
            units = "mm",
        )
    except KeyError:
            pass
    # overwriting any attributes for each variables
    set_attrs(dat.attrs, **kwargs)


def generate_cluster_netcdfs(h: HUC, series="ams", ndays=1):
    wt = cluster.cluster_weights(h, series, ndays)
    # to remove the "actual_range" attr
    wt["lat"] = 0 + wt.lat
    if wt.lon.min() > 180:
        # GIS compatible range
        wt["lon"] = wt.lon - 360
    dates = pd.read_csv(
        h.data_path(f"clusters-{series}_{ndays}day.csv"),
    ).groupby("cluster").end_date.agg(list).to_dict()
    precip = h.load_timeseries().prec
    for cl in range(len(wt.cluster)):
        c1 = wt.weights[:, :, cl].drop_vars("cluster")
        c1 = (c1 * c1.count() * 100).fillna(0).to_dataset()

        c1_dates = pd.to_datetime(pd.Series(dates[wt.cluster[cl].item()])) - pd.to_datetime("1915-1-1")
        c1_dates.index += 1
        c1_dates.index.name = "days"
        c1["date"] = xarray.DataArray(c1_dates.map(lambda x: x.days))

        c1_prec = precip.loc[dates[wt.cluster[cl].item()]].copy(deep=True)
        c1_prec.index = c1_dates.index
        c1["prec"] = xarray.DataArray(c1_prec)
        prepare_metadata(
            c1,
            f"HUC {h.huc_code}: {series} {ndays}day cluster{cl+1}"
        )
        print(f"exported-dss/{h.name.replace(' ', '')}_{series}{ndays}d_cluster-{cl+1}_prec-100mm.nc")
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

    dates = pd.read_csv(
        h.data_path(f"clusters-{series}_{ndays}day.csv"),
    ).end_date
    dates.index += 1
    dates.index.name = "days"
    precip = h.load_timeseries().prec[dates]
    precip.index = dates.index
    average_ds["date"] = xarray.DataArray((
        pd.to_datetime(dates) - pd.to_datetime("1915-1-1")
    ).map(lambda x: x.days))
    average_ds["prec"] = xarray.DataArray(precip)
    prepare_metadata(
        average_ds,
        f"HUC {h.huc_code}: {series} {ndays}day average"
    )
    print(f"exported-dss/{h.name.replace(' ', '')}_{series}{ndays}d_average_prec-100mm.nc")
    average_ds.to_netcdf(f"exported-dss/{h.name.replace(' ', '')}_{series}{ndays}d_average_prec-100mm.nc")


def generate_uniform_netcdf(h: HUC):
    wt = (h.weights.weights * h.weights.weights.count() * 100).to_dataset(name="weights")
    wt["lat"] = 0 + wt.lat
    if wt.lon.min() > 180:
        # GIS compatible range
        wt["lon"] = wt.lon - 360
    prepare_metadata(
        wt,
        f"HUC {h.huc_code}: uniform"
    )
    print(f"exported-dss/{h.name.replace(' ', '')}_uniform_prec-100mm.nc")
    wt.to_netcdf(f"exported-dss/{h.name.replace(' ', '')}_uniform_prec-100mm.nc")
