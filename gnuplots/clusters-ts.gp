set datafile separator ","
set datafile columnhead

set boxwidth 1
set style fill solid
unset xtics
unset ytics
set yrange [0:1]
set xrange [0:]

set palette maxcolors 7
set palette model RGB defined (1 "purple", 2 "forest-green", 3 "orange", 4 "blue", 5 "red", 6 "white", 7 "black")
set cbrange [1:8]
unset colorbox
set cbtics ('1' 1, '2' 2, '3' 3, '4' 4, '5' 5, '6' 6, '7' 7) offset screen 0,0.055

map_color(c)=c[3:]+0

HUCS = "01 02 03 04 05 06 07 08 10 11 12 13 14 15 16 17 18"
nhucs=words(HUCS)
set multiplot layout nhucs+1,1 spacing 0,0
# do for [huc in HUCS] {
#     set ylabel huc norotate
# plot "../data/output/".huc."/clusters-ams_1day.csv"\
#      using 0:(1):(stringcolumn("cluster")[3:3]+0) with boxes notitle fillcolor palette
# }

unset key
set rmargin screen 0.8
do for [h=1:nhucs] {
    huc=word(HUCS, h)
    if (h==nhucs){
	set colorbox user origin 0.85,0.1 size 0.05,0.8
    }
     set ylabel huc norotate
plot "../data/clusters-ams.csv"\
      using 0:(1):"H".huc with boxes notitle fillcolor palette
}
