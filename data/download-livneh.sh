rm -f livneh-files.txt
for yr in {1915..2011}; do echo "https://psl.noaa.gov/thredds/fileServer/Datasets/livneh/metvars/prec.${yr}.nc" >> livneh-files.txt; done;
parallel -j 4 curl -C - -o input/{0/} {0} < livneh-files.txt
