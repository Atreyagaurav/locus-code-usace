from src.huc import HUC
import sys
import xarray
from src.livneh import LivnehData
import src.precip as precip
from datetime import datetime as dt
from src.cluster import get_center
import json


huc = HUC(sys.argv[1])
series = sys.argv[2]
ndays = 1


if series == "ams":
    series_df = precip.load_ams_series(huc, ndays)
elif series == "pds":
    threshold = precip.get_threhold(huc, ndays)
    series_df = precip.load_pds_series(huc, ndays, threshold)
dates = [dt.fromisoformat(d) for d in series_df.end_date]

df = get_center(huc, dates)

series_df.index = dates
newdf = series_df.join(df)

center_data = [
    dict(
        date=r.end_date,
        precip=r.p_mm,
        x=r.x - 360,
        y=r.y,
    )
    for _,r in newdf.iterrows()
]

fname = huc.data_path(f"{series}_{ndays}day_dates.json")
print(fname)
with open(fname, "w") as w:
    json.dump(center_data, w)

with open(f"/tmp/{huc.huc_code}.txt", "w") as w:
    for d, day in newdf.end_date.items():
        w.write(f"gdaldem color-relief -alpha netcdf:data/output/{huc.huc_code}/prec-{d.year}.nc:prec precip-color-2.txt server/timeseries/{huc.huc_code}/{day}.png -b {d.dayofyear}\n")
