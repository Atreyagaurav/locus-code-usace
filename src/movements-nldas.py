import xarray
import pandas as pd
import numpy as np
import lmfit
import shapely
import regionmask
import matplotlib.pyplot as plt

from src.huc import HUC

h = HUC("02070002")
h = HUC("05")

data = xarray.load_dataset(f"data/nldas/{h.huc_code}/nldas-precip.nc")
region = regionmask.Regions([h.geometry])
mask = region.mask_3D_frac_approx(data.lon, data.lat)
weights = mask.where(mask>0, drop=True).isel(region=0).drop_vars(["abbrevs", "names", "region"])

clusters = xarray.load_dataset(h.data_path("clusters-weights_pds_1day.nc"))


matches = dict()

for cl in clusters.cluster:
    cl1 = clusters.weights.sel(cluster=cl).drop_vars("cluster")
    cl1["lon"] = cl1.lon - 360

    x = cl1.sel(lat=data.lat, lon=data.lon, method="nearest")
    x["lat"] = data.lat
    x["lon"] = data.lon

    diff = ((data.Rainf - x)**2).sum(dim=["lat", "lon"])
    
    matches[str(cl.values)] = diff.time.to_series().iloc[int(diff.argmin())]


plt.subplots_adjust(
    **{k: 0.06 for k in ["left", "bottom"]},
    **{k: 0.94 for k in ["top", "right"]},
    **{k: 0.2 for k in ["wspace", "hspace"]},
)
fig, axes = plt.subplots(
    nrows=1,
    ncols=len(matches),
    figsize=(5 * len(matches), 6),
    sharex=True,
    sharey=True,
    squeeze=True
)


for i, cl in enumerate(matches):
    x = data.Rainf.sel(time=matches[cl]).where(weights>0).plot(ax=axes[i])

plt.suptitle(f"Closest Matched Patterns in NLDAS2")
plt.savefig(h.image_path(f"{series}_{ndays}dy_nldas.png"))
print(":", h.image_path(f"{series}_{ndays}dy_nldas.png"))

from datetime import timedelta

for cl in matches:
    start = matches[cl] - timedelta(hours=12)
    end = matches[cl] + timedelta(hours=12)
    ind = (data.time > start) & (data.time < end)
    times = data.time.to_series().loc[ind.values]
    l = data.Rainf.sel(time=times.to_numpy()).where(weights>0)
    for t in times:
        data.Rainf.sel(time=t).where(weights>0).drop_vars("time").plot(
            vmin=float(l.min()), vmax=float(l.max()),
        )
        plt.savefig(f"/tmp/{t.isoformat()}")
        plt.close()
