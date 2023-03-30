from typing import List

import pandas as pd
from src.huc import HUC
from datetime import timedelta


def calculate_ams_series(huc: HUC, years: List[int], ndays: int) -> pd.DataFrame:
    basin = huc.rolling_timeseries(years, ndays)
    year = pd.DatetimeIndex(basin.index).year
    series = basin[basin == basin.groupby(year).transform(max)].dropna()
    ams_series = (
        pd.DataFrame({"p_mm": series.prec, "end_date": series.index})
        .reset_index()
        .drop(columns=["time"])
    )
    ams_series.loc[:, "duration"] = ndays
    ams_series.to_csv(huc.data_path(f"ams_{ndays}dy_series.csv"), index=None)
    print(":", huc.data_path(f"ams_{ndays}dy_series.csv"))
    return ams_series


def load_ams_series(huc: HUC, years: List[int], ndays: int) -> pd.DataFrame:
    try:
        return pd.read_csv(huc.data_path(f"ams_{ndays}dy_series.csv"))
    except FileNotFoundError:
        return calculate_ams_series(huc, years, ndays)


def calculate_pds_series(
    huc: HUC, years: List[int], ndays: int, threshold: float
) -> pd.DataFrame:
    basin = huc.rolling_timeseries(years, ndays)
    series = basin[basin.prec > threshold].dropna()
    pds_series = (
        pd.DataFrame({"p_mm": series.prec, "end_date": series.index})
        .reset_index()
        .drop(columns=["time"])
    )
    pds_series.loc[:, "duration"] = ndays
    pds_series.to_csv(huc.data_path(f"pds_{ndays}dy_series.csv"))
    print(":", huc.data_path(f"pds_{ndays}dy_series.csv"))
    return pds_series


def load_pds_series(
    huc: HUC, years: List[int], ndays: int, threshold: float
) -> pd.DataFrame:
    try:
        return pd.read_csv(huc.data_path(f"pds_{ndays}dy_series.csv"))
    except FileNotFoundError:
        return calculate_pds_series(huc, years, ndays, threshold)


def calculate_series_grids(huc: HUC, ams: pd.DataFrame) -> pd.DataFrame:
    def grids():
        for ind, row in ams.iterrows():
            end = pd.to_datetime(row.end_date)
            dates = [(end - timedelta(days=i)) for i in range(row.duration)]
            yield huc.get_gridded_df(dates).groupby("ids").prec.sum().to_dict()

    all_grids = pd.DataFrame(list(grids()), index=ams.end_date, dtype=float)
    return all_grids


def load_pds_grids(
    huc: HUC, years: List[int], ndays: int, threshold: float
) -> pd.DataFrame:
    try:
        return pd.read_csv(
            huc.data_path(f"pds_{ndays}dy_grids.csv"), index_col="end_date"
        )
    except FileNotFoundError:
        pds = load_pds_series(huc, years, ndays, threshold)
        grids = calculate_series_grids(huc, pds)
        grids.to_csv(huc.data_path(f"pds_{ndays}dy_grids.csv"))
        print(":", huc.data_path(f"pds_{ndays}dy_grids.csv"))
        return grids


def load_ams_grids(huc: HUC, years: List[int], ndays: int) -> pd.DataFrame:
    try:
        return pd.read_csv(
            huc.data_path(f"ams_{ndays}dy_grids.csv"), index_col="end_date"
        )
    except FileNotFoundError:
        ams = load_ams_series(huc, years, ndays)
        grids = calculate_series_grids(huc, ams)
        grids.to_csv(huc.data_path(f"ams_{ndays}dy_grids.csv"))
        print(":", huc.data_path(f"ams_{ndays}dy_grids.csv"))
        return grids
