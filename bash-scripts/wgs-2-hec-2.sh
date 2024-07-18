# Converts given nc raster file into a DSS grid file with 24 hr
# precipitation using SCS Type II weights
src_file=$1
river_name=`basename ${src_file%%_*}`
dss_file=${src_file%.*}.dss
zip_file=${src_file%.*}.zip
src_filename=`basename ${src_file}`
ext=${src_file##*.}

tmp_dir=`mktemp -d`
tmp_file=${tmp_dir}/${src_filename%.*}
warped_file=${tmp_file}-wrapped.${ext}

# whether it has 24 hrs or 48 hrs, actual time doesn't matter
# numbands=`ncdump $src_file | grep time | head -n 1 | awk -F'=' '{print $2}' | awk '{print $1}'`

numbands=24

im=0
for i in $(seq 1 $numbands); do
    warped_file=${tmp_file}-wrapped-$i.${ext}
    gdalwarp -s_srs WGS84 -t_srs hec-abers.prj -tr 2000 2000 -r average -b $i $src_file $warped_file
    start_time=`date -d "2011-07-31 + ${im} hours"  +"%d%^b%Y:%H%M"`
    end_time=`date -d "2011-07-31 + ${i} hours"  +"%d%^b%Y:%H%M"`
    asc_file=${tmp_dir}/SHG_${river_name}_PRECIP_${start_time}_${end_time}_Livneh.asc
    info_file=${tmp_dir}/SHG_${river_name}_PRECIP_${start_time}_${end_time}_Livneh.dssinfo
    gdal_translate -of aaigrid -a_nodata 0 netcdf:$warped_file $asc_file
    cp example.dssinfo $info_file
    (( im++ ))
done;
# exported-dss/NorthBranchPotomac_pds1d_cluster-1-nldas-2_prec-100mm.nc

# double the grids but with zero values
warped_file=${tmp_file}-wrapped-1.${ext}
calc_file=${tmp_file}-1.${ext}
calc_file_t=${tmp_file}-zeros.${ext}
gdal_calc.py --calc "a * 0" -a $warped_file --outfile $calc_file
gdal_translate -of aaigrid -a_nodata 0 netcdf:$calc_file $calc_file_t

im=0
for i in $(seq 1 $numbands); do
    start_time=`date -d "2011-07-31 + ${numbands} hours + ${im} hours"  +"%d%^b%Y:%H%M"`
    end_time=`date -d "2011-07-31 + ${numbands} hours + ${i} hours"  +"%d%^b%Y:%H%M"`
    asc_file=${tmp_dir}/SHG_${river_name}_PRECIP_${start_time}_${end_time}_Livneh.asc
    info_file=${tmp_dir}/SHG_${river_name}_PRECIP_${start_time}_${end_time}_Livneh.dssinfo
    cp $calc_file_t $asc_file
    cp example.dssinfo $info_file
    (( im++ ))
done;

csv2dss g $dss_file ${tmp_dir}/SHG_${river_name}_PRECIP*.asc
zip -j $zip_file ${tmp_dir}/SHG_${river_name}_PRECIP*.asc
zip -j $zip_file ${tmp_dir}/SHG_${river_name}_PRECIP*.prj
rm -r $tmp_dir
