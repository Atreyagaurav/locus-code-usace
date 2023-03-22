#!/usr/bin/env python
# coding: utf-8

# # Trinity Watershed
# 
# by: John Kucharksi | correspondance: johnkucharski@gmail.com | date: 05 March 2022
# The notebook analyzes the spatial patterns of extreme precipitation events in the Trinity Watershed, using the Livneh daily precipitation data and USGS 4 digit hydrologic unit code watershed boundary shapefile.$^{1, 2}$ </br>
#   
# References: <br>
# $^{1}$ Livneh B., E.A. Rosenberg, C. Lin, B. Nijssen, V. Mishra, K.M. Andreadis, E.P. Maurer, and D.P. Lettenmaier, 2013: A Long-Term Hydrologically Based Dataset of Land Surface Fluxes and States for the Conterminous United States: Update and Extensions, Journal of Climate, 26, 9384–9392. <br>
# $^{2}$ U.S. Geological Survey, 2020, National Waterboundary Dataset for 2 digit hydrologic Unit (HU) 02 (mid-atlantic), accessed April 11, 2020 at URL http://prd-tnm.s3-website-us-west-2.amazonaws.com/?prefix=StagedProducts/Hydrography/WBD/HU2/Shape/. ** 
# 
# **note: more information on the USGS National Hydrography program and products including the water boundary shape files, used in this analysis can be found here: https://www.usgs.gov/core-science-systems/ngp/national-hydrography/access-national-hydrography-products. 

# ## Data Processing
# The first half of this notebook, processes the raw livneh data.

# In[1]:


import os
import sys
from pathlib import Path 
from typing import List, Dict, Tuple
# projpath: str = f'{Path(os.path.abspath("")).resolve().parent.parent}/' #'/Users/johnkucharski/Documents/source/locus/'
projpath: str = "./"
sys.path.insert(0, projpath)

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import src.cluster as cluster
import src.livneh as livneh
from src.waterboundary import Waterboundary


# ### Folder Structure
# The data for this analysis are stored in the following directory structure:
# 
# data/ </br>
# &emsp;  |- input/ </br>
# &emsp;  |&emsp; |- livneh/ </br>
# &emsp;  |&emsp; |&emsp; '- prec.YEAR.nc (YEAR = 4-digit year [1915-2011])</br>
# &emsp;  |&emsp; '- waterboundary/  </br>
# &emsp;  |&emsp;  &emsp; '- WBD_XX_HU2_Shape/Shape/ (XX = 2-digit USGS HUC code) </br>
# &emsp;  |&emsp;  &emsp; &emsp;  |- WBDHUZZ.shp (ZZ = 1 or 2-digit HUC code size) </br>
# &emsp;  |&emsp;  &emsp; &emsp;  '- other related files (.shx, .cpg, .dbf, .sbn, etc.) </br>
# &emsp;  '- output/ </br>
# &emsp;   &emsp; '- trinity ** </br>
# &emsp;   &emsp;  &emsp; |- ams/ </br>
# &emsp;   &emsp;  &emsp; |&emsp;  '- Ndy_events.csv (N = duration in days) </br>
# &emsp;   &emsp;  &emsp; |&emsp;  '- Ndy_grids.csv  (N = duration in days) </br>
# &emsp;   &emsp;  &emsp; '- pds/ </br>
# &emsp;   &emsp;  &emsp; |&emsp;  '- Ndy_events.csv (N = duration in days) </br>
# &emsp;   &emsp;  &emsp; |&emsp;  '- Ndy_grids.csv (N = duration in days) </br>
# &emsp;   &emsp;  &emsp; '- prec.YEAR.csv (YEAR = 4-digit year [1915, 2011])
# 
# **note: to generalize this the folder structure between the outputs and the northbranch folder (a HUC08) should include folders with the HUC 2, 4, and 6 names.
# 
# 
# 

# ### Inputs

# In[2]:


inputs: str = f'{projpath}data/input/'


# The data being analyzed is daily gridded precipitation depths recorded from 1915 through 2011 in the North Branch of the Potomac Watershed in Western Maryland, USA. 
# The data for each day contains the preciptation depth in all 130 livneh grid cells (1/16th of a degree of latitude/longitude) located within or intersected by the North Branch of the Potomac 8 digit Hydrologic Unit Code (HUC08) boundary.

# ### Outputs

# In[3]:


output: str = f'{projpath}data/output/trinity/'


# The output data from this notebook are livneh files that collectively contain the partial duration series (PDS) from a peak over threshold analysis, that yields of days in the livneh record in which the rainfall exceeds the minimum annual maximum basin average value.

# In[4]:

from src.huc import HUC
huc_code: str = '1203'
huc = HUC(huc_code)
wbd = gpd.GeoDataFrame([huc.feature["properties"]], geometry=[huc.geometry], crs="EPSG:4326")
wbd_area = huc.areasqkm

# fpath: str = f'{inputs}waterboundary/'
# wbd = Waterboundary(path=fpath, code=huc_code).import_waterboundary()
# wbd_area = wbd.iloc[0].AreaSqKm
# wbd.head()


# Later the livneh grids are converted to polygons and clipped to the "wbd" watershed shape, this is a slow process. To speed it up first the netCDF livneh grids are clipped to a boundary box, created in the next cell.

# In[5]:


buffer: float = 1/32
# bbox = [west, south, east, north] coordinates
bbox: List[float] = list(huc.geometry.bounds)
bbox[0] = bbox[0] - buffer
bbox[1] = bbox[1] - buffer
bbox[2] = bbox[2] + buffer
bbox[3] = bbox[3] + buffer


# Gathers the list of input .NetCDF file path (as strings) to process, and the output paths (as strings) for the processed .csv files.

# In[6]:


srt_paths: List[str] = livneh.inputfilepaths(f'{inputs}', vars=['prec'], yrs=[i for i in range(1915, 2012)])
end_paths: List[str] = livneh.outputfilepaths(output, srt_paths)


# In[7]:


# ids = livneh.ids(srt_paths[0], boundarybox=bbox, wbd=wbd).drop(columns=['area_km2', 'area_weight'])
# print(ids.shape)
# ids.head()

# Processes the the livneh precipitation .NetCDF files returning processed .csv files, for the area of interest.

# In[8]:

# Took a lot of time, at least I removed some redundant parts
print_msg = livneh.process_files(srt_paths, end_paths, bbox, wbd)
exit(0)
# print(f'processed: {print_msg}')


# Takes the processed .csv livneh files for the area of interest, and returns partial duration (PDS) and annual maximum series (AMS) for the specified list of durations in days.

# In[9]:


print_msg = livneh.compute_series(end_paths, [1, 2, 3])
print(print_msg)


# In[10]:


df = pd.read_csv(f'{output}ams/1dy_grids.csv', index_col=[0])
print(df.shape)
df.head()


# In[11]:


df_clustered = cluster.storm_centers(df) 
print(df_clustered.shape)
df_clustered.head()


# In[12]:


ids = pd.read_csv(f'{output}ids.csv', index_col=[0])
ids = gpd.GeoDataFrame(ids, geometry=gpd.points_from_xy(ids['lat'], ids['lon']))
cluster_means = cluster.cluster_means(df_clustered, ids)
print(cluster_means.shape)
cluster_means.head()


# In[13]:


nclusters = sum([1 if type(i) is int else 0 for i in cluster_means.columns.values])
fig, axs = plt.subplots(nrows=nclusters, ncols=2, figsize=(20, 30), sharex=True, sharey=True)
for c in range(2):
    for i in range(nclusters):
        if c == 0:
           cluster_means.plot(ax=axs[i, c], column=i, vmin=20, vmax=110, legend = True) 
        else:
            cluster_means.plot(ax=axs[i, c], column=i, legend=True)
        #cluster_means.plot(ax=axs[cluster, column], column=cluster, vmin=20, vmax=110, legend = True) if column==0 else cluster_means.plot(ax=axs[cluster, column], column=cluster, legend=True)
        axs[i, c].set_title(f'cluster: {i}')


# In[14]:


norm = cluster.normalize_cluster(cluster_means, ids)
norm.head()

