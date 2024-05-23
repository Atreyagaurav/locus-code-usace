import pandas as pd
from src.huc import HUC
from datetime import timedelta


def calculate_ams_series(huc: HUC, ndays: int) -> pd.DataFrame:
    basin = huc.rolling_timeseries(ndays)
    # find the dry month for the water year cutoff
    months = pd.DatetimeIndex(basin.index).month
    m = basin.groupby(months).mean()
    cut_month = m.loc[m.prec==m.prec.min()].index[0]
    
    cut_days = [
        f"{y}-{cut_month:02d}-01" for y in
        pd.DatetimeIndex(basin.index).year.unique()
    ]
    blocks = basin.index.map(lambda d: d in cut_days).to_series().cumsum().to_list()
    series = basin[basin == basin.groupby(blocks).transform(max)].dropna()
    # year = pd.DatetimeIndex(basin.index).year
    # series = basin[basin == basin.groupby(year).transform(max)].dropna()
    ams_series = (
        pd.DataFrame({"p_mm": series.prec, "end_date": series.index})
        .reset_index()
        .drop(columns=["time"])
    )
    ams_series.loc[:, "duration"] = ndays
    ams_series.to_csv(huc.data_path(f"ams_{ndays}dy_series.csv"), index=None)
    print(":", huc.data_path(f"ams_{ndays}dy_series.csv"))
    return ams_series


def load_ams_series(huc: HUC, ndays: int) -> pd.DataFrame:
    try:
        return pd.read_csv(huc.data_path(f"ams_{ndays}dy_series.csv"))
    except FileNotFoundError:
        return calculate_ams_series(huc, ndays)


def calculate_pds_series(huc: HUC, ndays: int, threshold: float) -> pd.DataFrame:
    basin = huc.rolling_timeseries(ndays)
    series = basin[basin.prec > threshold].dropna()
    pds_series = (
        pd.DataFrame({"p_mm": series.prec, "end_date": series.index})
        .reset_index()
        .drop(columns=["time"])
    )
    pds_series.loc[:, "duration"] = ndays
    pds_series.to_csv(huc.data_path(f"pds_{ndays}dy_series.csv"), index=None)
    print(":", huc.data_path(f"pds_{ndays}dy_series.csv"))
    return pds_series


def load_pds_series(huc: HUC, ndays: int, threshold: float) -> pd.DataFrame:
    try:
        return pd.read_csv(huc.data_path(f"pds_{ndays}dy_series.csv"))
    except FileNotFoundError:
        return calculate_pds_series(huc, ndays, threshold)


def get_series_grids(huc: HUC, ams: pd.DataFrame) -> pd.DataFrame:
    def grids():
        for ind, row in ams.iterrows():
            end = pd.to_datetime(row.end_date)
            dates = [(end - timedelta(days=i)) for i in range(row.duration)]
            yield huc.get_gridded_df(dates).groupby("ids").prec.sum().to_dict()

    all_grids = pd.DataFrame(list(grids()), index=ams.end_date, dtype=float)
    return all_grids


def load_pds_grids(huc: HUC, ndays: int, threshold: float) -> pd.DataFrame:
    pds = load_pds_series(huc, ndays, threshold)
    return get_series_grids(huc, pds)


def load_ams_grids(huc: HUC, ndays: int) -> pd.DataFrame:
    ams = load_ams_series(huc, ndays)
    return get_series_grids(huc, ams)


def get_threhold(huc: HUC, ndays) -> float:
    ams = load_ams_series(huc, ndays)
    return ams.p_mm.min()
