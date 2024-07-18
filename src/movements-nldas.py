import xarray
import pandas as pd
import numpy as np
# import lmfit
import shapely
import regionmask
import matplotlib.pyplot as plt
import pytz
from datetime import timedelta, datetime

from src.huc import HUC
from src.livneh import LivnehData

h = HUC("02070002")
# h = HUC("05")
MATCH_NLDAS = True
series = "pds"
ndays = 1

data = xarray.load_dataset(f"data/nldas/{h.huc_code}/precip.nc")
region = regionmask.Regions([h.geometry])
region.mask_3D(data.lon, data.lat)
mask = region.mask_3D_frac_approx(data.lon, data.lat)
weights = mask.where(mask>0, drop=True).isel(region=0).drop_vars(["abbrevs", "names", "region"])

clusters = xarray.load_dataset(h.data_path(f"clusters-weights_{series}_{ndays}day.nc"))
clusters["lon"] = clusters.lon - 360


matches_liv = dict()
# matches based on nldas 24 hour rolling mean
if MATCH_NLDAS:
    precip_rol = data.Rainf.where(weights > 0).rolling({"time":23}, center=True).mean()
    precip_daily = precip_rol.sum(dim=["lat", "lon"])
    t = float(precip_daily.quantile(0.99))
    precips = precip_rol.where(precip_daily > t, drop=True)
    precip_norm = precips / precips.max(dim=["lat", "lon"])

    matches = dict()
    for cl in clusters.cluster:
        cl1 = clusters.weights.sel(cluster=cl).drop_vars("cluster")
        x = cl1.sel(lat=data.lat, lon=data.lon, method="nearest")
        x["lat"] = data.lat
        x["lon"] = data.lon
        x_norm = x / x.max()
        diff = ((precip_norm - x_norm)**2).max(dim=["lat", "lon"])
        mid = diff.time.to_series().iloc[int(diff.argmin())]
        start = mid - timedelta(hours=12)
        end = mid + timedelta(hours=12)
        matches[str(cl.values)] = [start, mid, end]
else:
    series_dates = pd.read_csv(h.data_path(f"{series}_{ndays}dy_series.csv")).end_date
    series_dates_nl = pd.to_datetime(series_dates.loc[series_dates > "1979-01-01"])
    series_dates_nl.name = "time"
    series_dates_nl = series_dates_nl.to_numpy()

    livneh = xarray.open_mfdataset(LivnehData.input_files_glob())
    livneh_huc = livneh.where(h.weights.weights > 0, drop=True).where(livneh.time >= data.time[0], drop=True).load()
    livneh_daily = livneh_huc.prec.sel(time=series_dates_nl)
    livneh_daily["lon"] = livneh_daily.lon - 360
    livneh_sum = livneh_daily.max(dim=["lat", "lon"])
    livneh_norm = livneh_daily / livneh_sum
    matches = dict()
    est = pytz.timezone("US/Eastern")
    for cl in clusters.cluster:
        cl1 = clusters.weights.sel(cluster=cl).drop_vars("cluster")
        x_norm = cl1 / cl1.max()
        diff = ((livneh_norm - x_norm)**2).sum(dim=["lat", "lon"])
        # adjust for timezone differences in the data
        matches_liv[str(cl.values)] = diff.time.to_series().index[int(diff.argmin())]
        start = diff.time.to_series().tz_localize(est).tz_convert(pytz.utc).tz_localize(None).index[int(diff.argmin())]
        mid = start + timedelta(hours=12)
        end = mid + timedelta(hours=12)
        matches[str(cl.values)] = [start, mid, end]


plt.subplots_adjust(
    **{k: 0.06 for k in ["left", "bottom"]},
    **{k: 0.94 for k in ["top", "right"]},
    **{k: 0.2 for k in ["wspace", "hspace"]},
)

if MATCH_NLDAS:
    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(matches),
        figsize=(5 * len(matches), 10),
        sharex=True,
        sharey=True,
        squeeze=True
    )
    for i, cl in enumerate(matches):
        precip_rol.sel(time=matches[cl][1]).where(weights > 0).plot(ax=axes[0, i])
        clusters.sel(cluster=cl).weights.plot(ax=axes[1, i])
else:
    fig, axes = plt.subplots(
        nrows=3,
        ncols=len(matches),
        figsize=(5 * len(matches), 15),
        sharex=True,
        sharey=True,
        squeeze=True
    )
    for i, cl in enumerate(matches):
        start, _, end = matches[cl]
        ind = (data.time >= start) & (data.time < end)
        times = data.time.to_series().loc[ind.values]
        data.Rainf.sel(time=times.to_numpy()).where(weights>0).sum(dim=["time"]).plot(ax=axes[0, i])
        ll = livneh_huc.prec.sel(time=matches_liv[cl]).drop_vars("time")
        ll["lon"] = ll.lon - 360
        ll.plot(ax=axes[1, i])
        clusters.sel(cluster=cl).weights.plot(ax=axes[2, i])
        

# livneh = xarray.open_mfdataset(LivnehData.input_files_glob())
# livneh_huc = livneh.where(h.weights.weights > 0, drop=True).where(livneh.time >= data.time[0], drop=True).load()
# livneh_huc["lon"] = livneh_huc.lon - 360

plt.suptitle(f"Closest Matched Patterns in NLDAS2")
plt.savefig(h.image_path(f"{series}_{ndays}dy_nldas{2 if MATCH_NLDAS else 1}.png"))
print(":", h.image_path(f"{series}_{ndays}dy_nldas{2 if MATCH_NLDAS else 1}.png"))
plt.close()

# replacing the data for only the data of the matches, temporary fix
# untill the whole database can be downloaded for the CONUS
data = xarray.load_dataset(f"data/nldas/nldas.nc")
region = regionmask.Regions([h.geometry])
mask = region.mask_3D_frac_approx(data.lon, data.lat)
weights = mask.where(mask>0, drop=True).isel(region=0).drop_vars(["abbrevs", "names", "region"])


precips = dict()
for cl in matches:
    start, mid, end = matches[cl]
    ind = (data.time >= start) & (data.time < end)
    times = data.time.to_series().loc[ind.values]
    l = data.Rainf.sel(time=times.to_numpy()).where(weights>0)
    precips[cl] = l
    # for t in times:
    #     data.Rainf.sel(time=t).where(weights>0).drop_vars("time").plot(
    #         vmin=float(l.min()), vmax=float(l.max()),
    #     )
    #     plt.savefig(f"/tmp/{cl}_{t.isoformat()}.png")
    #     plt.close()


scs = [0.011, 0.011, 0.012, 0.014, 0.015, 0.017, 0.018, 0.022, 0.027, 0.034, 0.054, 0.428, 0.109, 0.048, 0.034, 0.026, 0.023, 0.019, 0.016, 0.014, 0.012, 0.012, 0.012, 0.012]
for cl in precips:
    p = precips[cl].mean(dim=["lat", "lon"])
    p = p / p.sum()
    plt.plot(p.values, label=cl)
plt.plot(scs, linestyle='dotted', label="SCS Hydrograph")
plt.legend()
plt.savefig(h.image_path(f"{series}_{ndays}dy_nldas-hourly{2 if MATCH_NLDAS else 1}.png"))
plt.close()

# add a buffer of nan (which'll become 0 later) for now since there
# was a cropping mistake in processing
# newlat = precips["C-1"].lat.to_numpy()
# newlon = precips["C-1"].lon.to_numpy()

# newlat = (newlat[0] * 2 - newlat[1], newlat[-1] * 2 - newlat[-2])
# newlon = (newlon[0] * 2 - newlon[1], newlon[-1] * 2 - newlon[-2])
# for cl in precips:
#     d = precips[cl]
#     d = d.pad({"lat": 1, "lon":1})
#     d.lat.values[0] = newlat[0]
#     d.lat.values[-1] = newlat[1]
#     d.lon.values[0] = newlon[0]
#     d.lon.values[-1] = newlon[1]
#     precips[cl] = d


# time dimension is overwritten in the conversion to dss, so whatever
p_c1 = precips['C-1'].mean(dim=["lat", "lon"])
scs_c1 = xarray.DataArray(scs, coords=p.coords)

l = h.weights.weights.copy(deep=True)
l["lon"] = l.lon - 360
l["lat"].attrs = {'standard_name': 'latitude', 'long_name': 'latitude', 'units': 'degrees_north', 'axis': 'Y'}
l["lon"].attrs = {'units': 'degrees_east', 'standard_name': 'longitude', 'long_name': 'longitude'}
livneh_uniform_scs = scs_c1 * (l * 100 / l)
livneh_uniform_scs.to_netcdf(f'jenny/{h.name.replace(" ","")}_precip-100mm_1a.nc')

# mask1 = region.mask_3D_frac_approx(precips['C-1'].lon, precips['C-1'].lat)
# weights1 = mask1.where(mask1>0, drop=True).isel(region=0).drop_vars(["abbrevs", "names", "region"])
nldas_uniform_scs = scs_c1 * (weights * 100 / weights)
nldas_uniform_scs.to_netcdf(f'jenny/{h.name.replace(" ","")}_precip-100mm_1b.nc')

hydrographs = dict()
for cl in precips:
    p = precips[cl].mean(dim=["lat", "lon"])
    p = p / p.sum()
    scs2 = xarray.DataArray(scs, coords=p.coords)
    hydrographs[cl] = p
    livneh_uniform_clus = p * (l * 100 / l)
    livneh_uniform_clus.to_netcdf(f'jenny/{h.name.replace(" ","")}_precip-100mm_2a_c{cl[2:]}.nc')
    nldas_uniform_clus = p * (weights * 100 / weights)
    nldas_uniform_clus.to_netcdf(f'jenny/{h.name.replace(" ","")}_precip-100mm_2b_c{cl[2:]}.nc')


series_dates = pd.read_csv(h.data_path(f"{series}_{ndays}dy_series.csv")).end_date
series_dates_nl = pd.to_datetime(series_dates).to_numpy()

livneh = xarray.open_mfdataset(LivnehData.input_files_glob())
livneh_huc = livneh.where(h.weights.weights > 0, drop=True).load()
livneh_daily = livneh_huc.prec.sel(time=series_dates_nl)
livneh_daily["lon"] = livneh_daily.lon - 360
l = livneh_daily.mean(dim=["time"])
livneh_average_scs = scs_c1 * (l / l.mean() * 100)
livneh_average_scs.to_netcdf(f'jenny/{h.name.replace(" ","")}_precip-100mm_3.nc')


# livneh clusters + scs & nldas hourly hydrograph
for cl in precips:
    cl1 = clusters.weights.sel(cluster=cl).drop_vars("cluster")
    livneh_nldas = scs_c1 * (cl1 / cl1.mean() * 100)
    livneh_nldas.to_netcdf(f'jenny/{h.name.replace(" ","")}_precip-100mm_5_c{cl[2:]}.nc')
    livneh_nldas = hydrographs[cl] * (cl1 / cl1.mean() * 100)
    livneh_nldas.to_netcdf(f'jenny/{h.name.replace(" ","")}_precip-100mm_6_c{cl[2:]}.nc')

# nldas 24 hour mean + scs & nldas hourly hydrograph
for cl in precips:
    p = precips[cl].mean(dim=["time"])
    livneh_nldas = scs_c1 * (p / p.mean() * 100)
    livneh_nldas.to_netcdf(f'jenny/{h.name.replace(" ","")}_precip-100mm_7_c{cl[2:]}.nc')
    livneh_nldas = hydrographs[cl] * (p / p.mean() * 100)
    livneh_nldas.to_netcdf(f'jenny/{h.name.replace(" ","")}_precip-100mm_8_c{cl[2:]}.nc')


for cl in precips:
    l = precips[cl]
    # don't do sum first as nan will be converted to 0, then mean will be off
    cl_data = l / l.mean(dim=["lon", "lat"]).sum(dim=["time"]) * 100
    cl_data.to_netcdf(f'jenny/{h.name.replace(" ","")}_precip-100mm_10_c{cl[2:]}.nc')

pass
# for cl in precips:
#     p = precips[cl].mean(dim=["lat", "lon"])
#     fname = h.name.replace(" ","") + f"_{series}{ndays}d_cluster-{cl[2:]}-nldas{2 if MATCH_NLDAS else 1}_prec-100mm.nc"
#     # don't do sum first as nan will be converted to 0, then mean will be off
#     cl_data = l / l.mean(dim=["lon", "lat"]).sum(dim=["time"]) * 100
#     cl_data.to_netcdf("exported-dss/" + fname)


# for cl in precips:
#     p = precips[cl].mean(dim=["lat", "lon"])
#     p = p / p.sum()
#     scs2 = xarray.DataArray(scs, coords=p.coords)
#     # nldas mean with scs hydrograph
#     l = precip_rol.sel(time=mid).drop_vars("time").where(weights>0)
#     fname_scs = h.name.replace(" ","") + f"_{series}{ndays}d_cluster-{cl[2:]}-nldas-scs{2 if MATCH_NLDAS else 1}_prec-100mm.nc"
#     cl_data_mean = l / l.mean() * 100
#     (scs2 * cl_data_mean).to_netcdf("exported-dss/" + fname_scs)
#     # nldas mean with own hydrograph (reassign the magnitudes to average pattern)
#     fname_act = h.name.replace(" ","") + f"_{series}{ndays}d_cluster-{cl[2:]}-nldas-actual{2 if MATCH_NLDAS else 1}_prec-100mm.nc"
#     cl_data_mean = l / l.mean() * 100
#     (p * cl_data_mean).to_netcdf("exported-dss/" + fname_act)
#     # for livneh data
#     cl1 = clusters.weights.sel(cluster=cl).drop_vars("cluster").copy(deep=True)
#     cl_data_mean =  cl1 / cl1.mean() * 100
#     # with scs distribution
#     fname_livnehscs = h.name.replace(" ","") + f"_{series}{ndays}d_cluster-{cl[2:]}-livneh-scs{2 if MATCH_NLDAS else 1}_prec-100mm.nc"
#     (scs2 * cl_data_mean).to_netcdf("exported-dss/" + fname_livnehscs)
#     # livneh data with nldas's time distribution
#     fname_livnehnldas = h.name.replace(" ","") + f"_{series}{ndays}d_cluster-{cl[2:]}-livneh-nldas{2 if MATCH_NLDAS else 1}_prec-100mm.nc"
#     (p * cl_data_mean).to_netcdf("exported-dss/" + fname_livnehnldas)
#     # uniform distributions
#     fname_uniform_liv = h.name.replace(" ","") + f"_uniform_cluster-{cl[2:]}-livneh_prec-100mm.nc"
#     cl_data_uniform = cl1 / cl1 * 100
#     (p * cl_data_mean).to_netcdf("exported-dss/" + fname_uniform_liv)
#     fname_uniform_nldas = h.name.replace(" ","") + f"_uniform_cluster-{cl[2:]}-nldas_prec-100mm.nc"
#     cl_data_uniform = l / l * 100
#     (p * cl_data_mean).to_netcdf("exported-dss/" + fname_uniform_nldas)
    

# # uniform distributions with scs is same for any cluster
# fname_uniform_liv = h.name.replace(" ","") + f"_uniform-livneh_prec-100mm.nc"
# cl_data_uniform = cl1 / cl1 * 100
# (scs2 * cl_data_mean).to_netcdf("exported-dss/" + fname_uniform_liv)
# fname_uniform_nldas = h.name.replace(" ","") + f"_uniform-nldas_prec-100mm.nc"
# cl_data_uniform = l / l * 100
# (scs2 * cl_data_mean).to_netcdf("exported-dss/" + fname_uniform_nldas)


# #to only download the matched files form above. should ideally match
# #against the whole dataset.

# with open("nldas-list.txt") as r:
#     files = [l.strip() for l in r]


# with open("matched-files.txt", "w") as w:
#     for v in precips.values():
#         days = v.time.dt.strftime("https://data.gesdisc.earthdata.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.2.0/%Y/%j/NLDAS_FORA0125_H.A%Y%m%d.%H%M.020.nc").values
#         for d in days:
#             w.write(f"{d}\n")
