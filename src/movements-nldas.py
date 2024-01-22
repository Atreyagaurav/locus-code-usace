import xarray
import pandas as pd
import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.ndimage import generate_binary_structure, binary_erosion
import matplotlib.pyplot as plt
import lmfit
import shapely

from datetime import datetime as dt
from datetime import timedelta as td
import matplotlib.pyplot as plt

from src.huc import HUC

h = HUC("02070002")

def get_center(df):
    lat_lon = (len(df.index.levels[0]), len(df.index.levels[1]))
    grid = df.prec.to_numpy().reshape(lat_lon)
    basin = df.basin.to_numpy().reshape(lat_lon)
    # grid2 = gaussian_filter(grid, sigma=2, order=0)
    masked = np.multiply(grid, basin)
    center = np.where(masked == masked.max())
    x = df.index.levels[1][center[1][0]]
    y = df.index.levels[0][center[0][0]]

    model = lmfit.models.Gaussian2dModel()
    all_x = [x for _,x in df.index]
    all_y = [y for y,_ in df.index]
    all_z = masked.flatten()
    params = model.guess(all_z, all_x, all_y)
    return (params.get("centerx").value, params.get("centery").value)


data = xarray.load_dataset("nldas.nc")
data = data.rename_vars({"APCP": "prec"})

mask = xarray.DataArray(coords=data.isel(time=0).drop_vars("time").coords).to_dataframe(name="basin")

for ind in mask.index:
    pt = shapely.Point(*ind)
    mask.loc[ind] = h.geometry.contains(pt)
mask.basin = mask.basin.astype(int)

nyears = len(data.time) // 24

clusters = pd.read_csv(h.data_path(f"clusters-ams_1day.csv"), index_col="end_date").cluster.map(lambda c: c.split("-")[1]).to_dict()

for year in range(nyears):
    centers = pd.DataFrame(index=range(24), columns=["x", "y"])
    centers.index.name = "hour"
    for hour in range(24):
        ydata = data.isel(time = 24*year + hour)
        time = str(ydata.time.to_numpy()).split(':', maxsplit=1)[0]
        day, time = time.split('T')
        grid = ydata.drop_vars("time").to_dataframe()
        centers.loc[hour] = list(get_center(grid.join(mask)))
    centers.loc[:, "dx"] = centers.x.diff().shift(-1)
    centers.loc[:, "dy"] = centers.y.diff().shift(-1)
    centers.loc[:, "cluster"] = clusters[day]
    centers.dropna().to_csv(f"data/output/north-branch-potomac/nldas/{day}.csv")
