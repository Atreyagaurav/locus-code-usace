set datafile separator ","
set datafile columnhead

set xdata time
set timefmt "%Y-%m-%d"

set xtics out format "%Y"

huc="05"
prec_file="../data/output/".huc."/prec.csv"
ams_file="../data/output/".huc."/ams_1dy_series.csv"
pds_file="../data/output/".huc."/pds_1dy_series.csv"

set multiplot layout 2,1
set ylabel "Precipitation (mm/day)"
set xlabel "Time (year)"

plot prec_file u "time":"prec" w impulses lc "gray" title "Precipitation",\
     ams_file u "end_date": "p_mm" w points ls 7 title "AMS Series"
     

plot prec_file u "time":"prec" w impulses lc "gray" title "Precipitation",\
     pds_file u "end_date": "p_mm" w points ls 7 lc "orange" title "PDS Series"
     
