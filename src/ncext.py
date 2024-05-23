from src.huc import HUC
import sys
import xarray
from src.livneh import LivnehData


huc = HUC(sys.argv[1])
year = int(sys.argv[2])

if huc.weights is None:
    huc.calculate_weights()
netCDF = xarray.load_dataset(LivnehData.input_file(year))
prec = netCDF.sel(lat=huc.weights.lat, lon = huc.weights.lon)
prec = prec.where(huc.weights.weights>0, drop=True)
prec.encoding = netCDF.encoding
prec.to_netcdf(huc.data_path(f"prec-{year}.nc"))
