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
def add_metadata(dat: xarray.Dataset, name, **kwargs):
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

    # axis not required when we have standard name
    # dat.coords['lat'].attrs["axis"] = "Y"
    # dat.coords['lon'].attrs["axis"] = "X"
    set_attrs(
        dat.coords['lat'].attrs,
        standard_name = "grid_latitude",
        units = "degrees_north",
    )
    set_attrs(
        dat.coords['lon'].attrs,
        standard_name = "grid_longitude",
        units = "degrees_east",
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
    # overwriting any attributes for each variables
    set_attrs(dat.attrs, **kwargs)


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
    wt = h.weights.weights * h.weights.weights.count() * 100
    wt["lat"] = 0 + wt.lat
    if wt.lon.min() > 180:
        # GIS compatible range
        wt["lon"] = wt.lon - 360
    wt.to_netcdf(f"exported-dss/{h.name.replace(' ', '')}_uniform_prec-100mm.nc")
