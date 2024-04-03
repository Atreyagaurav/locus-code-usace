ls -1 server/download/*.nc | parallel --silent --bar gdaldem color-relief -alpha netcdf:{1}:weights precip-color.txt  'server/thumbnails/{1/.}.png'
