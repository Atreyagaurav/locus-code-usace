"""A command line entry point for the locus analysis"""
import argparse
import geopandas as gpd
import math
import matplotlib.pyplot as plt
import pandas as pd
import shapely
import time
from enum import Enum
from datetime import datetime
from typing import List

import src.cluster as cluster
import src.precip as precip
from src.huc import HUC


class CliAction(Enum):
    """Actions that can be performed in a basin with HUCode

    Each entries here should have curresponding methods in this module
    with the same name in lowercase that can be called with a huc.HUC
    object and commandline arguments.

    """
    CALCULATE_WEIGHTS = 0
    EXTRACT_ANNUAL_TIMESERIES = 1
    AMS_AND_PDS = 2
    FIND_CLUSTERS = 3
    PLOT_CLUSTERS = 4
    BATCH_PROCESS = 5


def calculate_weights(huc: HUC, args):
    huc.load_weights(calculate=True)
    return


def extract_annual_timeseries(huc: HUC, args):
    huc.load_timeseries(LivnehData.YEARS)
    return


def ams_and_pds(huc: HUC, args):
    ndays = args.num_days
    precip.load_ams_grids(huc, LivnehData.YEARS, ndays)
    threshold = _get_threhold(huc, ndays)
    precip.load_pds_grids(huc, LivnehData.YEARS, ndays, threshold)
    return


def find_clusters(huc: HUC, args):
    for series in args.series.split("+"):
        cluster.cluster_means(huc, series, args.num_days)


def plot_clusters(huc: HUC, args):
    ndays = args.num_days
    for series in args.series.split("+"):
        cluster_means = cluster.cluster_means(huc, series, ndays)
        clusters = [c for c in cluster_means.columns if c.startswith("C-")]
        nclusters = len(clusters)
        maximum = (
            math.ceil(cluster_means.loc[:, clusters].max().max() / 10) * 10
        )
        minimum = (
            math.floor(cluster_means.loc[:, clusters].min().min() / 10) * 10
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
        for i, clus in enumerate(sorted(clusters)):
            cluster_means.plot(
                ax=axs[0, i], column=clus, vmin=minimum, vmax=maximum, legend=True
            )
            cluster_means.plot(ax=axs[1, i], column=clus, legend=True)
            axs[0, i].set_title(f"cluster: {clus}")
            axs[1, i].set_title(f"cluster: {clus}")
        plt.suptitle(f"Precipitation Patterns in {huc}")
        plt.savefig(huc.image_path(f"{series}_{ndays}dy.png"))
        print(":", huc.image_path(f"{series}_{ndays}dy.png"))


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
        "-l",
        "--list-hucs",
        action="store_true",
        help="List HUCodes in the given category and exit"
    )
    parser.add_argument(
        "-d",
        "--details",
        action="store_true",
        help="print basin details before processing"
    )
    parser.add_argument(
        "-D",
        "--details-only",
        action="store_true",
        help="print basin details and exit"
    )
    parser.add_argument(
        "-s",
        "--series",
        choices=["ams", "pds", "ams+pds"],
        default="ams+pds",
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
    if args.list_hucs:
        for (code, name) in HUC.all_huc_codes(args.HUCode):
            print(code, ":", name)
        exit(0)

    huc = HUC(args.HUCode)
    print(f"** Basin: {huc}")
    print("Processed at", datetime.now())
    if args.details or args.details_only:
        print("Basin properties:")
        for (k, v) in huc.feature["properties"].items():
            print(f"    + {k} = {v}")
        if args.details_only:
            exit(0)

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
        print(f"*** {flag.name}")
        func_name = flag.name.lower()
        t1 = time.time()
        locals()[func_name](huc, args)
        dur = time.time() - t1
        print(f"    Time taken: {dur: .3f} seconds ({dur / 60: .2} minutes)")
        print(flush=True)
