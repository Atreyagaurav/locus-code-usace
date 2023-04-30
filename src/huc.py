import os
from typing import List, Iterator, Tuple
import itertools
import xarray
import fiona
import shapely
from src.livneh import LivnehData
from datetime import datetime
import pandas as pd
import geopandas as gpd

# https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/GPKG/WBD_National_GPKG.zip
# fiona.listlayers("./data/WBD_National_GPKG.gpkg")


class InvalidHUCode(Exception):
    "raised when the HUC code is invalid (not in the geopackage file)"
    pass


class HUC:
    """class for basins constructed based on their Hydrologic Unit Code

    The geometry is accessible through `geometry` property, and all
    the other properties of the feature in the geopackage is
    accessible with the dot notation, for example:

    ```
    h = HUC("1203")
    print(f"Name = {h.name}")
    print(f"Area = {h.areasqkm}")
    ```

    repr of the object is its name and the HUCode

    """
    GPKG_FILE: str = "./data/WBD_National_GPKG.gpkg"
    SOURCE_CRS: str = "EPSG:4326"
    AREA_CRS: str = "EPSG:3857"
    IDS_AND_WEIGHTS_FILENAME: str = "ids-and-weights.nc"

    @classmethod
    def all_huc_codes(cls, N: int) -> Iterator[Tuple[str, str]]:
        """List all HUCodes for the given category

        :param N: category of Hydrologic Unit (2, 4, 6, ..., 16)
        :type N: int
        :returns: Iterator[Tuple[str, str]]

        """
        with fiona.open(HUC.GPKG_FILE, layer=f"WBDHU{N}") as layer:
            yield from map(lambda l: (l["properties"][f"huc{N}"],
                                      l["properties"]["name"]), layer)

    def __init__(self, code: str):
        """Init HUC object

        :param code: HUCode for the basin
        :type code: str
        :returns:

        """
        N: int = len(code)
        if N % 2 == 1:
            raise InvalidHUCode(
                f"HUCode are even length, {code} has odd length")
        self.huc_code = code
        with fiona.open(HUC.GPKG_FILE, layer=f"WBDHU{N}") as layer:
            try:
                # the python filter is slow since it happens on the
                # python side. fiona 2.0 will have a new syntax
                # `l.filter(where="huc2='12'")` it does the filtering
                # from the SQL side and it is a lot faster. Link:
                # https://github.com/Toblerity/Fiona/issues/1016

                # if you do:
                # `pip install git+https://github.com/Toblerity/Fiona.git`
                # you can install the version 2 that's currently under
                # developement
                if fiona.__version__ > "2.":
                    self.feature = next(layer.filter(where=f"huc{N}='{code}'"))
                else:
                    self.feature = next(
                        filter(lambda f: f["properties"][f"huc{N}"] == code,
                               layer)
                    )
            except StopIteration:
                raise InvalidHUCode(f"No match for {code} in {HUC.GPKG_FILE}")
        self.geometry = shapely.geometry.shape(self.feature["geometry"])
        os.makedirs(self.data_path(), exist_ok=True)
        os.makedirs(self.image_path(), exist_ok=True)
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
        return f"{self.name} <HUC {self.huc_code}>"

    def geometry_as_geodataframe(self) -> gpd.GeoDataFrame:
        """geometry of the basin as geopandas geodataframe

        :returns:

        """
        return gpd.GeoDataFrame(
            [self.feature["properties"]], geometry=[
                self.geometry], crs=HUC.SOURCE_CRS
        )

    def buffered_bbox(self, buffer=LivnehData.RESOLUTION) -> List[float]:
        """buffered bounding box

        :param buffer: buffer distance around the geometry
        :type buffer: float
        :returns: list of buffered geometry bounds

        """
        bbox: List[float] = list(self.geometry.bounds)
        bbox[0] = bbox[0] - buffer
        bbox[1] = bbox[1] - buffer
        bbox[2] = bbox[2] + buffer
        bbox[3] = bbox[3] + buffer
        return bbox

    def data_path(self, filename: str = "") -> str:
        """path to a data file for this basin

        :param filename: name of the file
        :type filename: str
        :returns: relative path to the data file with given name

        """
        return os.path.join(f"./data/output/{self.huc_code}", filename)

    def image_path(self, filename: str = "") -> str:
        """path to the image file for this basin

        :param filename: filename of the image
        :type filename: str
        :returns: relative path to the image file

        """
        return os.path.join(f"./images/{self.huc_code}", filename)

    def load_timeseries(self) -> pd.DataFrame:
        """timeseries of the precipitation for given years.

        loads it using `load_annual_timeseries` for all the years and
        then concatenates the data

        :returns: dataframe with index 'time' containing iso formatted
                  dates and a column 'prec'

        """
        filename = self.data_path("prec.csv")
        if os.path.exists(filename):
            return pd.read_csv(filename, index_col="time")
        else:
            df = self.process_timeseries()
            df.to_csv(filename)
            return df

    def rolling_timeseries(self, ndays: int) -> pd.DataFrame:
        """same as load_timeseries but for rolling sum for multiple days

        :param ndays: number of days to collect the precipitation for
        :type ndays: int
        :returns: dataframe with index 'time' containing iso formatted dates
                  and a column 'prec'

        """
        return self.load_timeseries().rolling(ndays, min_periods=1).sum()

    def get_gridded_df(self, dates: List[datetime]) -> pd.DataFrame:
        """timeseries with each grids' values and weights for given dates

        :param dates: dates of interest
        :type dates: List[datetime]
        :returns: pandas dataframe with lat,lon as index and
                  time(iso date),ids,prec as column data

        """
        if self.weights is None:
            self.load_weights()
        year_julian = [(d.year, dates[0].timetuple().tm_yday) for d in dates]
        bbox = self.buffered_bbox()

        def get_df(yr, jul):
            netCDF = xarray.open_dataset(LivnehData.input_file(yr))
            lats = list(
                filter(
                    lambda il: il[1] > bbox[1] and il[1] < bbox[3],
                    enumerate(netCDF.lat.to_numpy()),
                )
            )
            lons = list(
                filter(
                    lambda il: il[1] > bbox[0] and il[1] < bbox[2],
                    enumerate(netCDF.lon.to_numpy() - 360),
                )
            )
            lats_ind = [i for i, l in lats]
            lons_ind = [i for i, l in lons]
            grid_day = netCDF.prec.isel(
                lon=lons_ind, lat=lats_ind, time=jul - 1
            ).fillna(0)
            grid_day["ids"] = self.weights.ids
            return grid_day.to_dataframe()

        return pd.concat(get_df(y, j) for y, j in year_julian)

    def process_timeseries(self) -> pd.DataFrame:
        """process the annual timeseries for a year and save it

        if weights for the LivnehData is unknown for the basin it'll
        run `calculate_weights`

        :returns: pandas dataframe with time (iso formatted date) as index and
                  prec column with precipitation

        """
        if self.weights is None:
            self.calculate_weights()

        def get_df(year):
            netCDF = xarray.open_dataset(LivnehData.input_file(year))
            weighted = (netCDF.prec * self.weights.weights).sum(dim=["lat", "lon"])
            weighted.name = "prec"
            return weighted.to_dataframe()

        return pd.concat(map(get_df, LivnehData.YEARS))

    def load_weights(self, /, calculate: bool = False):
        """load weights of cells in LivnehData for the basin

        :param calculate: whether to calculate the weights if it's not stored
                          locally (default False)
        :type calculate: bool
        :returns:

        """
        if os.path.exists(self.data_path(HUC.IDS_AND_WEIGHTS_FILENAME)):
            self.weights = xarray.open_dataset(
                self.data_path(HUC.IDS_AND_WEIGHTS_FILENAME)
            )
        else:
            if calculate:
                self.calculate_weights()
            else:
                self.weights = None

    def calculate_weights(self):
        """calculate and save the weights for LivnehData for this basin

        :returns:

        """
        netCDF = xarray.open_dataset(next(LivnehData.all_input_files()))
        bbox = self.buffered_bbox()
        lats = list(
            filter(
                lambda il: il[1] > bbox[1] and il[1] < bbox[3],
                enumerate(netCDF.lat.to_numpy()),
            )
        )
        lons = list(
            filter(
                lambda il: il[1] > bbox[0] and il[1] < bbox[2],
                enumerate(netCDF.lon.to_numpy() - 360),
            )
        )
        latlon = pd.Series(itertools.product(lats, lons))
        shift = LivnehData.RESOLUTION / 2

        geometry = [
            shapely.ops.clip_by_rect(self.geometry,
                                     lon - shift, lat - shift,
                                     lon + shift, lat + shift)
            for ((_, lat), (_, lon)) in latlon
        ]
        clipped = gpd.GeoDataFrame(
            index=latlon, geometry=geometry, crs=HUC.SOURCE_CRS)
        clipped = clipped.loc[clipped.geometry.map(lambda g: not g.is_empty)].to_crs(
            HUC.AREA_CRS
        )
        area: pd.Series = clipped.geometry.map(lambda g: g.area / 1_000_000)
        template = netCDF.prec.isel(time=0).drop_vars("time")
        self._set_and_save_weights(area, template=template)

    def _set_and_save_weights(self, area: pd.Series, /, template):
        areas = xarray.zeros_like(template)
        ids = xarray.zeros_like(template, dtype=int)
        i = 1
        for ((lat_ind, _), (lon_ind, _)), a in area.items():
            areas[(lat_ind, lon_ind)] = a
            ids[(lat_ind, lon_ind)] = i
            i += 1
        print(f"Number of Grid Cells in Basin: {i-1}")
        weights = xarray.Dataset()
        weights["ids"] = ids
        total_area = self.geometry_as_geodataframe().to_crs(
                HUC.AREA_CRS).loc[0, "geometry"].area / 1_000_000
        weights["weights"] = areas / total_area
        print(f"Sum of weights: {float(weights.weights.sum()): .3f}")
        self.weights = weights.where(weights.ids > 0, drop=True)
        self.weights.to_netcdf(self.data_path(HUC.IDS_AND_WEIGHTS_FILENAME))
        self.weights.to_dataframe().dropna().to_csv(
            self.data_path("ids.csv")
        )
        print(":", self.data_path(HUC.IDS_AND_WEIGHTS_FILENAME))
        print(":", self.data_path("ids.csv"))
