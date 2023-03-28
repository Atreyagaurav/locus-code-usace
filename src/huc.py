import xarray
import fiona
import shapely
from src.livneh import LivnehData
from datetime import datetime

# https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/GPKG/WBD_National_GPKG.zip
# fiona.listlayers("./data/WBD_National_GPKG.gpkg")

class InvalidHUCode(Exception):
    pass


class HUC:
    GPKG_FILE: str = "./data/WBD_National_GPKG.gpkg"
    SOURCE_CRS: str = "EPSG:4326"
    AREA_CRS: str = "EPSG:3857"
    IDS_AND_WEIGHTS_FILENAME: str = "ids-and-weights.nc"

    @classmethod
    def all_huc_codes(cls, N: int):
        with fiona.open(HUC.GPKG_FILE, layer=f"WBDHU{N}") as l:
            yield from map(lambda f: f["properties"][f"huc{N}"], l)

    def __init__(self, code):
        N: int = len(code)
        if N%2 == 1:
            raise InvalidHUCode(f"HUCode are even length, {code} has odd length")
        self.huc_code = code
        os.makedirs(self.data_path(), exist_ok=True)
        os.makedirs(self.image_path(), exist_ok=True)
        with fiona.open(HUC.GPKG_FILE, layer=f"WBDHU{N}") as l:
            try:
                # the python filter is slow since it happens on the
                # python side. fiona 2.0 will have a new syntax
                # `l.filter(where="huc2='12'")` it does the filtering
                # from the SQL side and it is a lot faster. Link:
                # https://github.com/Toblerity/Fiona/issues/1016
                
                # if you do: pip install git+https://github.com/Toblerity/Fiona.git
                # you can install the version 2 that's currently under developement
                if fiona.__version__ > "2.":
                    self.feature = next(l.filter(where=f"huc{N}='{code}'"))
                else:
                    self.feature = next(filter(lambda f: f["properties"][f"huc{N}"] == code, l))
            except StopIteration:
                raise InvalidHUCode(f"No match for {code} in {HUC.GPKG_FILE}")
        self.geometry = shapely.geometry.shape(self.feature["geometry"])
        self.load_weights(calculate=False)

    def __getattr__(self, obj):
        try:
            return super().__getattribute__(obj)
        except AttributeError as e:
            try:
                return self.feature["properties"][obj]
            except KeyError:
                raise AttributeError(e)

    def __repr__(self):
        return f'{self.name} <HUC {self.huc_code}>'

    def geometry_as_geodataframe(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame([self.feature["properties"]],
                                geometry=[self.geometry],
                                crs=HUC.SOURCE_CRS)
    
    def buffered_bbox(self, buffer=1/16):
        bbox: List[float] = list(self.geometry.bounds)
        bbox[0] = bbox[0] - buffer
        bbox[1] = bbox[1] - buffer
        bbox[2] = bbox[2] + buffer
        bbox[3] = bbox[3] + buffer
        return bbox

    def data_path(self, filename="") -> str:
        return os.path.join(f"./data/output/{self.huc_code}", filename)
    
    def image_path(self, filename="") -> str:
        return os.path.join(f"./images/{self.huc_code}", filename)

    def load_timeseries(self, years: List[int]) -> pd.DataFrame:
        return pd.concat(self.load_annual_timeseries(y)
                         for y in years)

    def rolling_timeseries(self, years: List[int], ndays: int) -> pd.DataFrame:
        return self.load_timeseries(years).rolling(ndays, min_periods=1).sum()

    def load_annual_timeseries(self, year: int) -> pd.DataFrame:
        filename = self.data_path(f"prec.{year}.csv")
        if os.path.exists(filename):
            return pd.read_csv(filename, index_col="time")
        else:
            return self.process_annual_timeseries(year)

    def get_gridded_df(self, dates: List[datetime]):
        if self.weights is None:
            self.load_weights()
        year_julian = [(d.year, dates[0].timetuple().tm_yday) for d in dates]
        bbox = self.buffered_bbox()
        def get_df(yr_jul):
            netCDF = xr.open_dataset(LivnehData.input_file(yr_jul[0]))
            lats = list(filter(lambda il: il[1] > bbox[1] and il[1] < bbox[3],
                               enumerate(netCDF.lat.to_numpy())))
            lons = list(filter(lambda il: il[1] > bbox[0] and il[1] < bbox[2],
                               enumerate(netCDF.lon.to_numpy() - 360)))
            lats_ind = [i for i, l in lats]
            lons_ind = [i for i, l in lons]
            grid_day = netCDF.prec.isel(lon=lons_ind,
                                        lat=lats_ind,
                                        time=yr_jul[1]-1).fillna(0)
            grid_day["ids"] = self.weights.ids
            return grid_day.to_dataframe()
        return pd.concat(get_df(yj) for yj in year_julian)

    def process_annual_timeseries(self, year: int) -> pd.DataFrame:
        if self.weights is None:
            self.calculate_weights()
        netCDF = xr.open_dataset(LivnehData.input_file(year))
        weighted = (netCDF.prec * self.weights.weights).sum(dim=["lat", "lon"])
        weighted.name = "prec"
        df = weighted.to_dataframe().set_index("time")
        df.to_csv(self.data_path(f"prec.{year}.csv"))
        return df

    def load_weights(self, /, calculate=False):
        if os.path.exists(self.data_path(HUC.IDS_AND_WEIGHTS_FILENAME)):
            self.weights = xarray.open_dataset(
                self.data_path(HUC.IDS_AND_WEIGHTS_FILENAME))
        else:
            if calculate:
                self.calculate_weights()
            else:
                self.weights = None

    def calculate_weights(self):
        netCDF = xr.open_dataset(next(LivnehData.all_input_files()))
        bbox = self.buffered_bbox()
        lats = list(filter(lambda il: il[1] > bbox[1] and il[1] < bbox[3],
                           enumerate(netCDF.lat.to_numpy())))
        lons = list(filter(lambda il: il[1] > bbox[0] and il[1] < bbox[2],
                           enumerate(netCDF.lon.to_numpy() - 360)))
        lats_ind = [i for i, l in lats]
        lons_ind = [i for i, l in lons]
        lats = [l for i, l in lats]
        lons = [l for i, l in lons]
        latlon = pd.Series(itertools.product(lats, lons))
        lat = latlon.map(lambda ll: ll[0]).to_numpy()
        lon = latlon.map(lambda ll: ll[1]).to_numpy()
        shift = 1/32
        n, s, e, w = lat + shift, lat - shift, lon + shift, lon - shift
        geometry = [Polygon(zip([w[i], e[i], e[i], w[i]],
                                [n[i], n[i], s[i], s[i]]))
                    for i in range(len(lat))]
        
        mask = self.geometry_as_geodataframe()
        geom_unclip = gpd.GeoDataFrame(
            index=itertools.product(lats_ind, lons_ind),
            geometry=geometry, crs=mask.crs)
        geom_unclip.set_geometry("geometry", inplace=True)
        # can be further sped up if we simplify the mask layer
        clipped = gpd.clip(geom_unclip,
                           mask,
                           keep_geom_type=False).to_crs(HUC.AREA_CRS)
        area = clipped.geometry.map(lambda g: g.area / 1_000_000)
        areas = xarray.zeros_like(netCDF.prec.isel(time=0).drop_vars("time"))
        ids = xarray.zeros_like(netCDF.prec.isel(time=0).drop_vars("time"),
                                dtype=int)

        i = 1
        for ll, a in area.items():
            areas[ll] = a
            ids[ll] = i
            i += 1
        # save the ids and weights
        weights = xarray.Dataset()
        weights["ids"] = ids
        weights["areas"] = areas
        weights["weights"] = areas / self.areasqkm
        self.weights = weights.where(weights.ids > 0, drop=True)
        self.weights.to_netcdf(self.data_path(HUC.IDS_AND_WEIGHTS_FILENAME))
        self.weights.ids.to_dataframe().dropna().to_csv(self.data_path("ids.csv"))

