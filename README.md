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
        Time taken:  2.290 seconds ( 0.038 minutes)
    *** EXTRACT_ANNUAL_TIMESERIES
        Time taken:  86.598 seconds ( 1.4 minutes)
    *** AMS_AND_PDS
        Time taken:  23.125 seconds ( 0.39 minutes)
	*** PLOT_CLUSTERS
	    Time taken:  3.401 seconds ( 0.057 minutes)

And then on the subsequent runs:

    *** CALCULATE_WEIGHTS
        Time taken:  0.002 seconds ( 2.9e-05 minutes)
    *** EXTRACT_ANNUAL_TIMESERIES
        Time taken:  0.075 seconds ( 0.0013 minutes)
    *** AMS_AND_PDS
        Time taken:  0.192 seconds ( 0.0032 minutes)
    *** PLOT_CLUSTERS
        Time taken:  3.503 seconds ( 0.058 minutes)

For North Branch Potomac (HUCode: 02070002) first runtime are as follows:

    *** CALCULATE_WEIGHTS
        Time taken:  0.104 seconds ( 0.0017 minutes)
    *** EXTRACT_ANNUAL_TIMESERIES
        Time taken:  83.067 seconds ( 1.4 minutes)
    *** AMS_AND_PDS
        Time taken:  9.011 seconds ( 0.15 minutes)
    *** PLOT_CLUSTERS
        Time taken:  2.308 seconds ( 0.038 minutes)

For Mid Atlantic Region <HUC 02>

    *** CALCULATE_WEIGHTS
        Time taken:  64.774 seconds ( 1.1 minutes)
    *** EXTRACT_ANNUAL_TIMESERIES
        Time taken:  91.243 seconds ( 1.5 minutes)
    *** AMS_AND_PDS
        Time taken:  22.159 seconds ( 0.37 minutes)
	*** PLOT_CLUSTERS
        Time taken:  22.557 seconds ( 0.38 minutes)
	
Seems like the extraction part is similar for all the basins, but the other ones vary by size. But it's reasonable time if you want to process a lot of basins.


# Tips and tricks
## Running in parallel
The original plan was to run everything without multiprocessing in the module itself, so multiple basins could be processed in parallel, but there seems to be some problem with the reading of netCDF from xarray or something that is giving some troubles. will have to look at it.

The plan is to use the `--list-hucs` command to get a list of codes for all basins, and then run the batch processing in parallel for them.

## Remore extra paddings for plots
The plots are a little weird, and I'm not that familiar with matplotlib, it seems to have a lot of extra border specially since we don't know how many clusters will be there, so I use `imagemagick` to trim the borders.

Here this example command will remove the excess padding and then add 20x20 padding on all sides

    mogrify -trim +repage -bordercolor white -border 20x20  images/02/*
## Generating reports
The output of the command can be piped, only the reporting texts from the main function of the `src/locus.py` is output to the `stdout`. The piped contents are in `emacs-org` format, which can be exported to any other formats (html,latex,pdf,etc) with emacs, if you do not have emacs you can use `pandoc` to convert it. Or you can modify the code to print markdown syntax instead, the output is simple so you only have to change all `"*"` in headings to `"#"`.

If you need to print any debugging information that's not supposed to goto the report, print it in the `stderr`.

# Note
The time reported can include other steps than the one you asked it to do, if that is needed. For example, if you asked for extracting the timeseries, then it'll calculate the weights if there is no weights, or if you ask for `ams/pds` it'll extract the timeseries.
