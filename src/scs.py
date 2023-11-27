from src.huc import HUC
import src.cluster as cluster
import pandas as pd
import numpy as np
import xarray

# # Did this in bash
# scs_typeII = pd.Series([
#     0.000, 0.011, 0.022, 0.034, 0.048, 0.063, 0.080, 0.098,
#     0.120, 0.147, 0.181, 0.235, 0.663, 0.772, 0.820, 0.854, 0.880,
#     0.903, 0.922, 0.938, 0.952, 0.964, 0.976, 0.988, 1.000
# ])

# fractions = scs_typeII.diff().dropna()
# fractions.index.name = "time"
# frac = xarray.DataArray(fractions)

hus = ["1203", "02070002", "05"]
for h in hus:
    h = HUC(h)
    wt = cluster.cluster_weights(h, "pds", 1)
    # to remove the "actual_range" attr
    wt["lat"] = 0 + wt.lat
    if wt.lon.min() > 180:
        # GIS compatible range
        wt["lon"] = wt.lon - 360
    for cl in range(len(wt.cluster)):
        c1 = wt.weights[:, :, cl].drop_vars("cluster")
        c1 = (c1 * c1.count()).fillna(0).to_dataset()
        c1.to_netcdf(f"exported-dss/{h.huc_code}_cluster-{cl+1}_prec-1mm.nc")

# reprojection needs to be done with qgis for now.

# c1_proj = xarray.load_dataset("/tmp/processing_EBhPjb/661580c1e0e045fea11999cf1e73881e/OUTPUT.nc")

# prec = xarray.Dataset(coords=c1_proj.coords)
# for t,w in fractions.items():
#     prec[f"time{t}"] = w * c1_proj.Band1.fillna(0.0)

# prec.to_netcdf(h.data_path("cluster-1_weighted-prec-hec.nc"))
