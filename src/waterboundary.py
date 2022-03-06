from typing import List
from dataclasses import dataclass
from geopandas import gpd
from pathlib import Path

@dataclass
class Waterboundary:
    '''
    A dataclass for data to access and import USGS waterboundary data shapefiles.
    '''
    crs: str = '4326'
    code: str = '02070002'
    path: str =  '' # /Users/johnkucharski/Documents/source/locus/data/input/waterboundary/'
    
    def filepath(self) -> str:
        return f'{self.path}WBD_{self.code[0:2]}_HU2_Shape/Shape/WBDHU{len(self.code)}.shp'
    def import_file(self):
        return gpd.read_file(self.filepath()).to_crs(f'EPSG:{self.crs}')
    def import_waterboundary(self) -> gpd.GeoDataFrame:
        wbd: gpd.GeoDataFrame = self.import_file()
        return wbd[wbd[f'HUC{len(self.code)}'] == self.code]
    def area(self) -> float:
        wbd = self.import_waterboundary()
        return wbd.AreaSqKm.iloc[0]
        