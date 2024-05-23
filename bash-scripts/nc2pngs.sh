filename=$1
year=$2
outdir=$3

numbands=`ncdump $filename | head -n 3 | tail -n+3 | awk -F'(' '{print $2}' | awk '{print $1}'`
for i in $(seq 1 $numbands); do
    day=`date -d "${year}-01-01 + ${i} days - 1 day"  +"%Y-%m-%d"`
    gdaldem color-relief -alpha netcdf:$filename:prec precip-color-2.txt ${outdir}/${day}.png -b $i
done;
