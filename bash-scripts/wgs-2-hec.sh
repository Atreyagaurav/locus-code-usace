# Converts given nc raster file into a DSS grid file with 24 hr
# precipitation using SCS Type II weights
src_file=$1
river_name=`echo $2 | tr '[:lower:]' '[:upper:]'`
dss_file=${src_file%.*}.dss
src_filename=`basename ${src_file}`
ext=${src_file##*.}
tmp_dir=${src_file%.*}.d
mkdir -p $tmp_dir
# tmp_dir=`mktemp -d`
tmp_file=${tmp_dir}/${src_filename%.*}
warped_file=${tmp_file}-wrapped.${ext}


fractions=(0.011 0.011 0.012 0.014 0.015 0.017 0.018 0.022 0.027 0.034 0.054 0.428 0.109 0.048 0.034 0.026 0.023 0.019 0.016 0.014 0.012 0.012 0.012 0.012)

gdalwarp -s_srs WGS84 -t_srs hec-abers.prj -dstnodata 0 -tr 2000 2000 -r average $src_file $warped_file

count=0
count2=1
for frac in ${fractions[@]}; do
    asc_file=${tmp_dir}/SHG_${river_name}_PRECIP_31JUL2011:`printf '%02d' $count`00_31JUL2011:`printf '%02d' $count2`00_RADAR.asc 
    calc_file=${tmp_file}-${frac}.${ext}
    if [ ! -f "$calc_file" ]; then
        gdal_calc.py --calc "a * ${frac} * 50" -a $warped_file --outfile $calc_file
    fi
    gdal_translate -of aaigrid netcdf:$calc_file $asc_file
    (( count++ ))
    (( count2++ ))
done;
csv2dss g $dss_file ${tmp_dir}/SHG_${river_name}_PRECIP*.asc
# rm -r $tmp_dir
