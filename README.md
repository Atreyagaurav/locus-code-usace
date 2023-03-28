# locus

Identifies spatial patterns in extreme precipitation events.

## Preparing Datasets
### Precipitation Data
Livneh datasets can be downloaded from: https://psl.noaa.gov/thredds/catalog/Datasets/livneh/metvars/catalog.html

Example link: https://psl.noaa.gov/thredds/fileServer/Datasets/livneh/metvars/prec.1915.nc
Data is available from 1915 to 2011.

There is a download script (bash) that'll download the files for you in `data` directory, you need gnu parallel and curl for downloads, if you don't then it'll generate the links in the file `data/livneh-files.txt`, you can download it using your favourite download manager.

### Water Boundary Data
Download the watershed boundaries from [USGS TNM Download (v2.0)](https://apps.nationalmap.gov), [direct link](https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/GPKG/WBD_National_GPKG.zip).

It's 4.8 GB to download and 7.1 GB after you unzip it.

## Run Times
This is the runtime on Trinity (HUCode: 1203) on my laptop (CPU: AMD Ryzen 7 6800U with Radeon Graphics (16) @ 2.700GHz).

On the first run, where it has to calculate everything:

    *** CALCULATE_WEIGHTS
    *** Time taken:  2.290 seconds ( 0.038 minutes)
    *** EXTRACT_ANNUAL_TIMESERIES
    *** Time taken:  86.598 seconds ( 1.4 minutes)
    *** AMS_AND_PDS
    *** Time taken:  23.125 seconds ( 0.39 minutes)
	*** PLOT_CLUSTERS
	*** Time taken:  3.401 seconds ( 0.057 minutes)

And then on the subsequent runs:

    *** CALCULATE_WEIGHTS
    *** Time taken:  0.002 seconds ( 2.9e-05 minutes)
    *** EXTRACT_ANNUAL_TIMESERIES
    *** Time taken:  0.075 seconds ( 0.0013 minutes)
    *** AMS_AND_PDS
    *** Time taken:  0.192 seconds ( 0.0032 minutes)
    *** PLOT_CLUSTERS
    *** Time taken:  3.503 seconds ( 0.058 minutes)

For North Branch Potomac (HUCode: 02070002) first runtime are as follows:

    *** CALCULATE_WEIGHTS
    *** Time taken:  0.104 seconds ( 0.0017 minutes)
    *** EXTRACT_ANNUAL_TIMESERIES
    *** Time taken:  83.067 seconds ( 1.4 minutes)
    *** AMS_AND_PDS
    *** Time taken:  9.011 seconds ( 0.15 minutes)
    *** PLOT_CLUSTERS
    *** Time taken:  2.308 seconds ( 0.038 minutes)
