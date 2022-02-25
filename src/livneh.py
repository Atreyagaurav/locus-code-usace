from datetime import timedelta
from typing import List
from dataclasses import dataclass

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

import multiprocessing as mp


def inputfilepaths(directory: str, vars: List[str], yrs: List[int]) -> List[str]:
    '''
    Generates a list of string file paths in the specified directory with the form <var>.<year>.nc

    Args: 
        directory (str): directory path. 
        vars (List[str]): list of variable identification strings (i.e. prec, tmin, tmax, etc.).
        yrs (List[int]): list of year strings in the form yyyy.
    
    Returns: 
        List[str]: file paths
    '''
    return [f'{directory}{str(v)}.{str(yr)}.nc' for yr in yrs for v in vars]

def outputfilepaths(directory: str, filelist: List[str], extension: str = '.csv') -> List[str]:
    '''
    Generates a list of file paths in the form <var>.<year>.<extension>

    Args:
        directory (str): directory path.
        filelist (List[str]): list of paths to files that will be processed.
        extension (str): string output file extension, .csv by default.
    
    Returns: 
        List[str]: file paths
    '''
    return [f'{directory}{path.rsplit("/", 1)[1].rsplit(".", 1)[0]}{extension}' for path in filelist]

def process_files(inputpaths: List[str], outputpaths: List[str], boundarybox: List[int], wbd: gpd.GeoDataFrame):
    '''
    Multithreaded livneh precipitation file procesessing, returning .csv for each year
    
    Args:
        inputpaths (List[str]): list string of file paths to livneh precipitation NetCDF files.
        outputpaths (List[str]): list of string file paths for the processed .csv files.
        boundarybox (List[int]): 180 degree decimal degree latitute and longitude edges creating a spatial box from which the Livneh data is extracated and processed.
        wbd (geopandas.GeoDataFrame): spatial dataframe containing the watershed polygon from which Livneh data is extracted and processed.
    
    Returns:
        Writes a string containing the processed file names. 
    
    Note:
        **Also writes the processed .csv files out.
    '''
    results = []
    df = import_file(inputpaths[0], boundarybox, wbd)
    ids = grid_ids_and_areas(df, wbd.iloc[0].AreaSqKm)
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap_async(process_file, [(inputpaths[i], outputpaths[i], boundarybox, wbd, ids) for i in range(len(inputpaths))]).get()
    pool.close()
    ids.to_csv(f'{outputpaths[0].rsplit("/", 1)[0]}/ids.csv')
    return results

def ids(inputpath: str, boundarybox: List[int], wbd: gpd.GeoDataFrame, crs: int = 4326):
    df = import_file(inputpath, boundarybox, wbd)
    df = grid_ids_and_areas(df, wbd.iloc[0].AreaSqKm)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat, crs=f'EPSG:{str(crs)}'))

def process_file(inputpath: str, outputpath: str, bbox: List[float], wbd: gpd.GeoDataFrame, ids:gpd.GeoDataFrame) -> None:
    '''
    Chains the import_file and grids_ids_and_areas
    '''
    df = import_file(inputpath, boundingbox = bbox, mask = wbd)
    df =  df.merge(ids, on=['lat', 'lon'], how='inner')
    df.to_csv(outputpath, index=False)   
    return outputpath.rsplit('/', 1)[1]

def import_file(filepath: str, boundingbox: List[int], mask: gpd.GeoDataFrame)-> pd.DataFrame:
    '''
    Imports a Livneh NetCDF file and returns a geopandas GeoDataFrame containing the Livneh data clipped to the mask area.

    Args:
        filepath (str): Livneh NetCDF string file path
        boundingbox (List[int]): list of latitude and longitude coordinates in form [E, S, W, N] corresponding with the maximum extents of the mask.
        mask (geopandas.GeoDataFrame): a geopandas data frame containing the region to which the Livneh data is clipped.
        
    Returns: 
        gpd.GeoDataFrame: containing ['date', 'prec', 'lat' and 'lon'] for polygons constructed the ['lat', 'lon'] centroids of each Livneh gridcell.
    '''
    #Import NetCDF file as XArray data
    netCDF = xr.open_dataset(filepath)
    if boundingbox is not None:
        # bounding box is based on 180 longitudes
        # livneh data is based on 360 degree longitude   
        netCDF = netCDF \
        .where(netCDF.lon > boundingbox[0] + 360, drop=True) \
        .where(netCDF.lat > boundingbox[1], drop=True) \
        .where(netCDF.lon < boundingbox[2] + 360, drop=True) \
        .where(netCDF.lat < boundingbox[3], drop=True)
    df: pd.DataFrame = netCDF.to_dataframe().reset_index()
    df['date'] = pd.to_datetime(df.time)
    df.drop(columns=['time'], inplace = True)
    # Geometry (fix lon, make polygons from centroid lats and lons, geopandas)
    df.lon = df.lon.apply(lambda x: x - 360)
    lat, lon, shift = df.lat.to_numpy(), df.lon.to_numpy(), 1/32
    n, s, e, w = lat + shift, lat - shift, lon + shift, lon - shift
    geometry = [Polygon(zip([w[i], e[i], e[i], w[i]], [n[i], n[i], s[i], s[i]])) for i in range(len(lat))]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=mask.crs)
    return gpd.clip(gdf, mask, keep_geom_type=False)

def grid_ids_and_areas(df: gpd.GeoDataFrame, wbd_area: float):
    '''
    Creates a dataframe containing the gridcells geometries, areas and ids.
    
    Args:
        df (geopandas.GeoDataFrame): A geopandas dataframe containing the Livneh gridcell data of interest, clipped to a watershed boundary.
        wbd_area (float): the size of the watershed of interest.
        
    Returns:
        geopandas.GeoDataFrame containing ['id', 'lat', 'lon', 'area_km2', 'area_weight'] where 
            * id is the gridcell id, 
            * lat and lon are the centroid coordinates, 
            * area_km2 is the gridcell area (clipped to the waterboundary) in sqkm, 
            * area_weight is the portion of the watershed contained in gridcell.
    '''
    # don't need 365 copies, just 1 day
    df = df[df.date == df.loc[df.index[0], 'date']].copy(deep=True)
    df.sort_values(by=['lat', 'lon'], ascending=[False, True], inplace=True)
    df.reset_index(inplace=True)
    df['id'] = df.index.to_numpy()
    df.to_crs('EPSG:3857', inplace=True)
    df['area_km2'] = df.geometry.area.to_numpy() / 1000000
    df['area_weight'] = df.area_km2.to_numpy() / wbd_area
    #df = df.filter(['id', 'lat', 'lon', 'area_km2', 'area_weight']) 
    return df.filter(['id', 'lat', 'lon', 'area_km2', 'area_weight']) 

def compute_series(outpaths: List[str], ndays: List[int]):
    '''
    Multithreaded compute for the partial duration (PDS) and annual maximum series (AMS) for a specified set of durations (in days), writes the series data out to .csv files containing: (1) the series events dates, (2) the series gridded data.
    
    Args:
        outpaths (List[str]): List of string paths for the processed .csv files for each PDS or AMS.
        ndays (List[int]): List of integer durations in days.
        
    Returns:
        A string indicating sucess (or failure) for each duration.
        
    Note:
        **Also writes out .csv files: (1) summarizing the series events (dates, overall depth, etc.), (2) the series gridded data. 
    '''
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap_async(_ams_and_pds, [(outpaths, n) for n in ndays]).get()
    pool.close()
    return results

def _ams_and_pds(outpaths: List[str], ndays: int):
    ams_data = ams(outpaths, ndays)
    pds_data = pds(outpaths, np.min(ams_data[0].p_mm.to_numpy()), ndays)
    return 'success'

def ams(outpaths: List[str], ndays: int = 1) -> pd.DataFrame:
    '''
    Computes the Annual Maximum Series from the set of processed annual livneh data files identified by outpaths.
    
    Args:
        outpaths (List[str]): a list of string paths to processed annual livneh data files.
        ndays (int): the n day event window over which the AMS is computed.
        
    Return:
        pandas.DataFrame: containing the AMS
        
    Note: Also writes out the AMS series (amsNdy.csv) and gridded event data (amsNdy_grids.csv) where N = ndays, to an 'ams' sub-directory in the outpaths directory.
    '''
    # import all processed livneh data and get basin average precipitation data
    dfs = series_data(outpaths, ndays)
    # find ams series
    df_ams = dfs[0]
    df_ams['yr'] = pd.DatetimeIndex(df_ams.date).year
    df_ams = df_ams[df_ams.p_mm == df_ams.groupby(['yr']).p_mm.transform(max)].reset_index()
    df_ams = df_ams.drop(columns=['index']).rename(columns={'date': 'end_date'})    
    df_ams['start_date'] = pd.to_datetime(df_ams.loc[:,'end_date'].to_numpy()) - pd.to_timedelta(ndays - 1, unit='d')
    # print ams data
    ams_grids = event_data(df_ams, dfs[1])
    df_ams.to_csv(f'{outpaths[0].rsplit("/", 1)[0]}/ams/{str(ndays)}dy_events.csv', index=False)
    ams_grids.to_csv(f'{outpaths[0].rsplit("/", 1)[0]}/ams/{str(ndays)}dy_grids.csv')
    return df_ams, ams_grids

def pds(outpaths: List[str], threshold: float, ndays:int = 1):
    # import all processed livneh data and get basin average precipitation data
    dfs = series_data(outpaths, ndays)
    # find peaks over theshold series
    df_pds = dfs[0]
    df_pds = df_pds[df_pds.p_mm >= threshold]
    df_pds = df_pds.rename(columns={'date': 'end_date'})
    df_pds['start_date'] = pd.to_datetime(df_pds.loc[:,'end_date'].to_numpy()) - pd.to_timedelta(ndays - 1, unit='d')
    # print pds data
    pds_grids = event_data(df_pds, dfs[1])
    df_pds.to_csv(f'{outpaths[0].rsplit("/", 1)[0]}/pds/{str(ndays)}dy_events.csv', index=False)
    pds_grids.to_csv(f'{outpaths[0].rsplit("/", 1)[0]}/pds/{str(ndays)}dy_grids.csv')
    return df_pds, pds_grids    

def series_data(outpaths: List[str], ndays: int) -> pd.DataFrame:
    # import all processed livneh data
    df_all = gpd.GeoDataFrame()
    for path in outpaths:
        df_all = pd.read_csv(path) if df_all.empty else df_all.append(pd.read_csv(path), ignore_index=True)
    # basin average precipitation data
    df_basin = df_all
    df_basin['p_mm'] = df_basin.prec.to_numpy() * df_basin.area_weight.to_numpy()
    df_basin = df_basin[['date', 'p_mm']].groupby(['date']).sum().reset_index()
    df_basin.p_mm = df_basin.p_mm.rolling(ndays, min_periods=1).sum()
    return df_basin, df_all

def event_data(series: pd.DataFrame, all_grids: pd.DataFrame) -> pd.DataFrame:    
    events: List[pd.Series] = []
    for _, row in series.iterrows():
        event_days: pd.DataFrame = all_grids[(pd.to_datetime(row.start_date) <= pd.to_datetime(all_grids.date)) & (pd.to_datetime(all_grids.date) <= pd.to_datetime(row.end_date))].sort_values(by=['id']).filter(['id', 'p_mm']).rename(columns={'p_mm': row.start_date})
        event_days = event_days.groupby(['id']).sum().T
        events.append(event_days)
    series_grids = pd.DataFrame().append(events)
    return series_grids   