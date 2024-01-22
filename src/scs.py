from src.huc import HUC
import src.cluster as cluster
import pandas as pd
import numpy as np
import xarray


# # Did this in bash
scs_typeII = pd.Series([
    0.000, 0.011, 0.022, 0.034, 0.048, 0.063, 0.080, 0.098,
    0.120, 0.147, 0.181, 0.235, 0.663, 0.772, 0.820, 0.854, 0.880,
    0.903, 0.922, 0.938, 0.952, 0.964, 0.976, 0.988, 1.000
])

fractions = scs_typeII.diff().dropna()
fractions.index.name = "time"
frac = xarray.DataArray(fractions)

hus = [("1203", 2), ("02070002", 1)]
for (h, ndays) in hus:
    h = HUC(h)
    for series in ["ams", "pds"]:
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
        csum = pd.read_csv(
            h.data_path(f"clusters-summary-{series}_{ndays}day.csv"), index_col="cluster"
        ).loc[:, "count"]
        cweights = xarray.DataArray.from_series(csum / csum.sum())
        average = (wt.weights * cweights).sum(["cluster"])
        average_ds = (
            average * average.where(average>0).count() * 100
        ).to_dataset(name="weights")
        average_ds.to_netcdf(f"exported-dss/{h.name.replace(' ', '')}_{series}{ndays}d_average_prec-100mm.nc")
    temp = c1.where(c1.weights.fillna(0)==0).fillna(100)
    temp.to_netcdf(f"exported-dss/{h.name.replace(' ', '')}_{series}{ndays}d_uniform_prec-100mm.nc")


data = xarray.load_dataset(f"exported-dss/{h.name.replace(' ', '')}_{series}{ndays}d_uniform_prec-100mm.nc")
# prec = xarray.Dataset(coords=c1_proj.coords)
# for t,w in fractions.items():
#     prec[f"time{t}"] = w * c1_proj.weights.fillna(0.0)
# prec.to_netcdf("nbpotomac-cluster1.nc")

# reprojection needs to be done with qgis for now.

# c1_proj = xarray.load_dataset("/tmp/processing_EBhPjb/661580c1e0e045fea11999cf1e73881e/OUTPUT.nc")

# prec = xarray.Dataset(coords=c1_proj.coords)
# for t,w in fractions.items():
#     prec[f"time{t}"] = w * c1_proj.Band1.fillna(0.0)

# prec.to_netcdf(h.data_path("cluster-1_weighted-prec-hec.nc"))
