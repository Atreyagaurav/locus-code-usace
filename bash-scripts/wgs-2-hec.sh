# Converts given nc raster file into a DSS grid file with 24 hr
# precipitation using SCS Type II weights
src_file=$1
# first part before the _ is the name
river_name=`basename ${src_file%%_*}`
dss_file=${src_file%.*}.dss
zip_file=${src_file%.*}.zip
src_filename=`basename ${src_file}`
ext=${src_file##*.}
# tmp_dir=${src_file%.*}.d
# mkdir -p $tmp_dir
tmp_dir=`mktemp -d`
tmp_file=${tmp_dir}/${src_filename%.*}
warped_file=${tmp_file}-wrapped.${ext}


frac_zeros=(0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)
# for potomac 1 day event
case $river_name in
    NorthBranchPotomac)
	frac21=(0.011 0.011 0.012 0.014 0.015 0.017 0.018 0.022 0.027 0.034 0.054 0.428 0.109 0.048 0.034 0.026 0.023 0.019 0.016 0.014 0.012 0.012 0.012 0.012)
	offset=(1 2 3 4)
	;;
    Trinity)
	frac21=(0.0049 0.0050 0.0051 0.0053 0.0055 0.0058 0.0060 0.0063 0.0066 0.0069 0.0073 0.0077 0.0101 0.0107 0.0115 0.0124 0.0136 0.0150 0.0201 0.0228 0.0267 0.0379 0.0501 0.1121)
	frac22=(0.2407 0.0754 0.0429 0.0294 0.0246 0.0213 0.0159 0.0142 0.0130 0.0119 0.0111 0.0104 0.0080 0.0075 0.0071 0.0067 0.0064 0.0061 0.0059 0.0056 0.0054 0.0052 0.0050 0.0049)
	offset=(2 3 4 5 6 7 8)
	;;
    *)
	echo "Unknown River: Precipitation distribution factors not available"
	exit 1
	;;
esac

# # for trinity 48 hour event


gdalwarp -s_srs WGS84 -t_srs hec-abers.prj -tr 2000 2000 -r average $src_file $warped_file

count=0
count2=1
for frac in ${frac21[@]}; do
    asc_file=${tmp_dir}/SHG_${river_name}_PRECIP_31JUL2011:`printf '%02d' $count`00_31JUL2011:`printf '%02d' $count2`00_Livneh.asc 
    info_file=${tmp_dir}/SHG_${river_name}_PRECIP_31JUL2011:`printf '%02d' $count`00_31JUL2011:`printf '%02d' $count2`00_Livneh.dssinfo 
    calc_file=${tmp_file}-${frac}.${ext}
    if [ ! -f "$calc_file" ]; then
        gdal_calc.py --calc "a * ${frac}" -a $warped_file --outfile $calc_file
    fi
    gdal_translate -of aaigrid -a_nodata 0 netcdf:$calc_file $asc_file
    cp example.dssinfo $info_file
    (( count++ ))
    (( count2++ ))
done;

if [[ -v frac22 ]]; then
    count=0
    count2=1
    for frac in ${frac22[@]}; do
	asc_file=${tmp_dir}/SHG_${river_name}_PRECIP_01AUG2011:`printf '%02d' $count`00_01AUG2011:`printf '%02d' $count2`00_Livneh.asc 
	calc_file=${tmp_file}-${frac}.${ext}
	if [ ! -f "$calc_file" ]; then
            gdal_calc.py --calc "a * ${frac}" -a $warped_file --outfile $calc_file
	fi
	gdal_translate -of aaigrid -a_nodata 0 netcdf:$calc_file $asc_file
	(( count++ ))
	(( count2++ ))
    done;
fi

for day in ${offset[@]}; do
    count=0
    count2=1
    for frac in ${frac_zeros[@]}; do
	asc_file=${tmp_dir}/SHG_${river_name}_PRECIP_0${day}AUG2011:`printf '%02d' $count`00_0${day}AUG2011:`printf '%02d' $count2`00_Livneh.asc 
	calc_file=${tmp_file}-${frac}.${ext}
	if [ ! -f "$calc_file" ]; then
            gdal_calc.py --calc "a * ${frac}" -a $warped_file --outfile $calc_file
	fi
	gdal_translate -of aaigrid -a_nodata 0 netcdf:$calc_file $asc_file
	(( count++ ))
	(( count2++ ))
    done;
done;
csv2dss g $dss_file ${tmp_dir}/SHG_${river_name}_PRECIP*.asc
zip -j $zip_file ${tmp_dir}/SHG_${river_name}_PRECIP*.asc
zip -j $zip_file ${tmp_dir}/SHG_${river_name}_PRECIP*.prj
rm -r $tmp_dir
