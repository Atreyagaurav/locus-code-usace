from typing import List
import itertools
import xarray

import os.path
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
    return [f'{directory}{v}.{yr}.nc' for yr in yrs for v in vars]


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
    def new_path(old_path):
        bn = os.path.basename(old_path)
        no_ext, _ = os.path.splitext(bn)
        return os.path.join(directory, f"{no_ext}{extension}")

    return [new_path(path) for path in filelist]


def process_files(inputpaths: List[str],
                  outputpaths: List[str],
                  boundarybox: List[int],
                  wbd: gpd.GeoDataFrame):
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
    ids, areas = grid_ids_and_areas(inputpaths[0], boundarybox, wbd)
    pool = mp.Pool(mp.cpu_count() - 1) # leave one so that your computer doesn't freeze
    results = pool.starmap_async(process_file,
                                 [(inputpaths[i],
                                   outputpaths[i],
                                   ids, areas, wbd.iloc[0].areasqkm)
                                  for i in range(len(inputpaths))]).get()
    pool.close()
    return results


def process_file(inputpath: str,
                 outputpath: str,
                 ids,
                 areas,
                 wbd_area) -> None:
    '''
    Chains the import_file and grids_ids_and_areas
    '''
    df = import_file(inputpath, ids, areas, wbd_area)
    print(f"to csv: {outputpath}")
    df.to_csv(outputpath)


def import_file(filepath: str, ids, areas, wbd_area) -> pd.DataFrame:
    '''
    Imports a Livneh NetCDF file and returns a geopandas GeoDataFrame containing the Livneh data clipped to the mask area.

    Args:
        filepath (str): Livneh NetCDF string file path
        wbd_area (float): the size of the watershed of interest.

    Returns:
        gpd.GeoDataFrame: containing ['date', 'prec', 'lat' and 'lon'] for polygons constructed the ['lat', 'lon'] centroids of each Livneh gridcell.
    '''
    netCDF = xr.open_dataset(filepath)
    netCDF["ids"] = ids
    netCDF["area_km2"] = areas
    netCDF["area_weight"] = areas / wbd_area
    df = netCDF.where(netCDF.area_weight > 0, drop=True).to_dataframe()
    df.index.names = ["date", "lat", "lon"]
    return df.dropna()


def grid_ids_and_areas(filepath: str, bbox: List[int], mask: gpd.GeoDataFrame):
    '''
    Creates a dataframe containing the gridcells geometries, areas and ids.

    Args:
        filepath (str): Livneh NetCDF string file path
        bbox (List[int]): list of latitude and longitude coordinates in form [E, S, W, N] corresponding with the maximum extents of the mask.
        mask (geopandas.GeoDataFrame): a geopandas data frame containing the region to which the Livneh data is clipped.

    Returns:
        tuple of ids and areas
    '''
    #Import NetCDF file as XArray data
    netCDF = xr.open_dataset(filepath)

    if bbox is not None:
        # b box is based on 180 longitudes
        # livneh data is based on 360 degree longitude
        lats = list(filter(lambda il: il[1] > bbox[1] and il[1] < bbox[3],
                           enumerate(netCDF.lat.to_numpy())))
        lons = list(filter(lambda il: il[1] > bbox[0] and il[1] < bbox[2],
                           enumerate(netCDF.lon.to_numpy() - 360)))
        lats_ind = [i for i, l in lats]
        lons_ind = [i for i, l in lons]
        lats = [l for i, l in lats]
        lons = [l for i, l in lons]
    else:
        lats = netCDF.lats.to_list()
        lons = netCDF.lons.to_list()
        # Geometry (fix lon, make polygons from centroid lats and lons, geopandas)
    # temporary fix to not have it do all this processing for each
    # file. since I assume they use the same grid.
    latlon = pd.Series(itertools.product(lats, lons))
    lat = latlon.map(lambda ll: ll[0]).to_numpy()
    lon = latlon.map(lambda ll: ll[1]).to_numpy()
    shift = 1/32
    n, s, e, w = lat + shift, lat - shift, lon + shift, lon - shift
    geometry = [Polygon(zip([w[i], e[i], e[i], w[i]],
                            [n[i], n[i], s[i], s[i]]))
                for i in range(len(lat))]
    geom_unclip = gpd.GeoDataFrame(
        index=itertools.product(lats_ind, lons_ind),
        geometry=geometry, crs=mask.crs)
    geom_unclip.set_geometry("geometry", inplace=True)
    clipped = gpd.clip(geom_unclip,
                       mask,
                       keep_geom_type=False).to_crs("EPSG:3857")
    area = clipped.geometry.map(lambda g: g.area / 1_000_000)
    areas = xarray.zeros_like(netCDF.prec.isel(time=0).drop_vars("time"))
    ids = xarray.zeros_like(netCDF.prec.isel(time=0).drop_vars("time"))
    i = 0
    for ll, a in area.iteritems():
        areas[ll] = a
        ids[ll] = i
        i += 1
    return ids, areas


def compute_series(outpaths: List[str], ndays: List[int]):
    print("compute_series")
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
    pool = mp.Pool(1)# mp.cpu_count()-1) # leave one so that your computer doesn't freeze
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

    basin, grids = series_data(outpaths, ndays) # basin avg precip and gridded data for all yrs
    basin['yr'] = pd.DatetimeIndex(basin['date']).year
    series = basin[basin['p_mm'] == basin.groupby(['yr'])['p_mm'].transform(max)].reset_index() # annual max series
    series = series.drop(columns=['index']).rename(columns={'date': 'end_date'})
    series['start_date'] = pd.to_datetime(series.loc[:, 'end_date'].to_numpy()) - pd.to_timedelta(ndays - 1, unit='d')
    series_grids = event_data(series, grids)
    series.to_csv(f'{outpaths[0].rsplit("/", 1)[0]}/ams/{str(ndays)}dy_events.csv', index=False)
    series_grids.to_csv(f'{outpaths[0].rsplit("/", 1)[0]}/ams/{str(ndays)}dy_grids.csv')
    return series, series_grids


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
    df_pds = dfs[0].copy(deep=True)
    df_pds = df_pds[df_pds.p_mm >= threshold]
    df_pds = df_pds.rename(columns={'date': 'end_date'})
    df_pds['start_date'] = pd.to_datetime(df_pds.loc[:,'end_date'].to_numpy()) - pd.to_timedelta(ndays - 1, unit='d')
    # print pds data
    pds_grids = event_data(df_pds, dfs[1].copy(deep=True))
    df_pds.to_csv(f'{outpaths[0].rsplit("/", 1)[0]}/pds/{str(ndays)}dy_events.csv', index=False)
    pds_grids.to_csv(f'{outpaths[0].rsplit("/", 1)[0]}/pds/{str(ndays)}dy_grids.csv')
    return df_pds, pds_grids


def series_data(outpaths: List[str], ndays: int) -> pd.DataFrame:
    # import all processed livneh data
    dfs: List[pd.DataFrame] = []
    for path in outpaths:
        dfs.append(pd.read_csv(path))
    df = pd.concat(dfs)
    basin = df.copy(deep = True)
    basin['p_mm'] = basin['prec'].to_numpy() * basin['area_weight'].to_numpy()
    basin = basin[['date', 'p_mm']].groupby(['date']).sum().reset_index().copy(deep=True)
    basin['p_mm'] = basin['p_mm'].rolling(ndays, min_periods=1).sum()
    return basin, df

    # df_all = gpd.GeoDataFrame()
    # for path in outpaths:
    #     df_all = pd.read_csv(path) if df_all.empty else df_all.append(pd.read_csv(path), ignore_index=True)
    # # basin average precipitation data
    # df_basin = df_all.copy(deep=True)
    # df_basin['p_mm'] = df_basin.prec.to_numpy() * df_basin.area_weight.to_numpy()
    # df_basin = df_basin[['date', 'p_mm']].groupby(['date']).sum().reset_index()
    # df_basin.p_mm = df_basin.p_mm.rolling(ndays, min_periods=1).sum()
    # return df_basin, df_all


def event_data(series: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    dfs: List[pd.Series] = []
    df['date'] = pd.to_datetime(df['date'])
    for _, row in series.iterrows():
        event = df.loc[df['date'].between(pd.to_datetime(row['start_date']), pd.to_datetime(row['end_date']))]
        #event = df[(pd.to_datetime(row['start_date']) <= pd.to_datetime(df['date'])) & (pd.to_datetime(df['date']) <= pd.to_datetime(row['end_date']))].copy(deep=True)
        #event = event.sort_values(by = ['id']).filter(['id', 'p_mm']).rename({'p_mm': row['start_date']}).copy(deep=True)
        event = event.sort_values(by=['id']).filter(['id', 'prec']).rename(columns={'prec': row['start_date']}).copy(deep=True)
        event = event.groupby(['id']).sum().T
        dfs.append(event)
    return pd.concat(dfs)
    # events: List[pd.Series] = []
    # for _, row in series.iterrows():
    #     event_days: pd.DataFrame = all_grids[(pd.to_datetime(row.start_date) <= pd.to_datetime(all_grids.date)) & (pd.to_datetime(all_grids.date) <= pd.to_datetime(row.end_date))].sort_values(by=['id']).filter(['id', 'p_mm']).rename(columns={'p_mm': row.start_date})
    #     event_days = event_days.groupby(['id']).sum().T
    #     events.append(event_days)
    # series_grids = pd.DataFrame().append(events)
    # return series_grids
