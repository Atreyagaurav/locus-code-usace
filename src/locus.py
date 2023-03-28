"""A command line entry point for the locus analysis"""
import argparse
import geopandas as gpd
import math
import matplotlib.pyplot as plt
import pandas as pd
import shapely
import time
from enum import Enum

import src.cluster as cluster
import src.precip as precip
from src.huc import HUC


YEARS = list(range(1915, 2011))


class CliAction(Enum):
    CALCULATE_WEIGHTS = 0
    EXTRACT_ANNUAL_TIMESERIES = 1
    AMS_AND_PDS = 2
    PLOT_CLUSTERS = 3
    BATCH_PROCESS = 4


def calculate_weights(huc: HUC, args):
    huc.load_weights(calculate=True)
    return


def extract_annual_timeseries(huc: HUC, args):
    huc.load_timeseries(YEARS)
    return


def ams_and_pds(huc: HUC, args):
    ndays = args.num_days
    precip.load_ams_grids(huc, YEARS, ndays)
    threshold = _get_threhold(huc, ndays)
    precip.load_pds_grids(huc, YEARS, ndays, threshold)
    return


def _get_threhold(huc: HUC, ndays) -> float:
    ams = precip.load_ams_series(huc, YEARS, ndays)
    return ams.p_mm.min()


def plot_clusters(huc: HUC, args):
    ndays = args.num_days
    series = args.series
    if series == "ams":
        grids = precip.load_ams_grids(huc, YEARS, ndays)
    elif series == "pds":
        threshold = _get_threhold(huc, ndays)
        grids = precip.load_pds_grids(huc, YEARS, ndays, threshold)
    elif series == "both":
        args.series = "ams"
        plot_clusters(huc, args)
        args.series = "pds"
        plot_clusters(huc, args)
        return
    else:
        return

    df_clustered = cluster.storm_centers(grids)
    ids = pd.read_csv(huc.data_path("ids.csv"), index_col="ids")
    shift = 1 / 32
    geometry = [
        shapely.box(lon - shift, lat - shift, lon + shift, lat + shift)
        for lat, lon in zip(ids["lat"], ids["lon"])
    ]
    ids = gpd.GeoDataFrame(ids, geometry=geometry)
    ids.set_index(pd.Index(ids.index, dtype=int), inplace=True)
    means = df_clustered.groupby("cluster").mean().T
    cluster_means = ids.join(
        means.set_index(pd.Index(means.index.map(float), dtype=int))
    )
    nclusters = len(df_clustered.cluster.unique())
    maximum = (
        math.ceil(cluster_means.loc[:, list(
            range(nclusters))].max().max() / 10) * 10
    )
    minimum = (
        math.floor(cluster_means.loc[:, list(
            range(nclusters))].min().min() / 10) * 10
    )

    plt.subplots_adjust(
        **{k: 0.01 for k in ["left", "bottom"]},
        **{k: 0.99 for k in ["top", "right"]},
        **{k: 0.1 for k in ["wspace", "hspace"]},
    )
    fig, axs = plt.subplots(
        nrows=2,
        ncols=nclusters,
        figsize=(5 * nclusters, 12),
        sharex=True,
        sharey=True
    )
    for i in range(nclusters):
        cluster_means.plot(
            ax=axs[0, i], column=i, vmin=minimum, vmax=maximum, legend=True
        )
        cluster_means.plot(ax=axs[1, i], column=i, legend=True)
        axs[0, i].set_title(f"cluster: {i}")
        axs[1, i].set_title(f"cluster: {i}")
    plt.savefig(huc.image_path(f"{series}_{ndays}dy.png"))
    print(huc.image_path(f"{series}_{ndays}dy.png"))


def cli_parser():
    parser = argparse.ArgumentParser(
        prog="locus",
        description="Analyse extreme precipitation pattern",
        epilog="https://github.com/Atreyagaurav/locus-code-usace",
    )
    parser.add_argument("HUCode")
    parser.add_argument(
        "-n",
        "--num-days",
        default=1,
        type=int,
        help="Number of Days for ams/pds event"
    )
    parser.add_argument(
        "-s",
        "--series",
        choices=["ams", "pds", "both"],
        default="both",
        help="ams or pds event to plot",
    )
    for act in CliAction:
        flag_name = act.name.lower().replace("_", "-")
        parser.add_argument(
            f"-{flag_name[0]}",
            f"--{flag_name}",
            action="store_true",
            help=f"run the {act.name.lower()} function",
        )
    return parser


if __name__ == "__main__":
    parser = cli_parser()
    args = parser.parse_args()
    huc = HUC(args.HUCode)
    print(f"Processing for: {huc}")
    if args.batch_process:
        flags = sorted(
            filter(lambda f: f != CliAction.BATCH_PROCESS, CliAction),
            key=lambda a: a.value,
        )
    else:
        flags = sorted(
            filter(lambda f: args.__dict__.get(f.name.lower()), CliAction),
            key=lambda a: a.value,
        )
    if len(flags) == 0:
        parser.print_help()
        exit(0)
    for flag in flags:
        func_name = flag.name.lower()
        t1 = time.time()
        locals()[func_name](huc, args)
        dur = time.time() - t1
        print(f"*** {flag.name}")
        print(f"*** Time taken: {dur: .3f} seconds ({dur / 60: .2} minutes)")
