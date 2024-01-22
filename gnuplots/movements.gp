reset
set datafile separator ","
unset key

set xtics 0.25 rotate format "%.2f"
set ytics 0.25 format "%.2f"
set grid
set multiplot layout 2,2

hucs="1203 02070002"
ndays="2 1"
series="pds"

do for [i=1:2]{
    HUC=word(hucs, i)
    nday=word(ndays, i)
    set title sprintf("HUC: %s (%s day)", HUC, nday)
    plot "../data/output/".HUC."/basin.csv"u ($1+360):2\
	 with lines lw 2 lc "black" notitle,\
	 "../data/output/".HUC."/".series."_".nday."dy_vectors.csv"\
	 using "x1":"y1":"ux1":"uy1":"cluster" with vectors\
	 head size graph 0.01,30 lc var lw 0.5 title "d-1",\
	 "" using "x2":"y2":"ux2":"uy2":"cluster" with vectors\
	 head size graph 0.01,30 lc var lw 0.5 title "d+1",\
	 "" using "x2":"y2":"cluster" with points ls 7\
	 lc var ps 0.8 title "d"
}

do for [i=1:2]{
    HUC=word(hucs, i)
    nday=word(ndays, i)
    set title sprintf("HUC: %s (%s day)", HUC, nday)
    plot "../data/output/".HUC."/basin.csv"u ($1+360):2\
	 with lines lw 2 lc "black" notitle,\
	 "../data/output/".HUC."/".series."_".nday."dy_vectors_mean.csv"\
	 using "x2":"y2":"cluster" with points ls 7\
	 lc var ps 1.6 title "mean(d)",\
	 "" using "x2":"y2":"ux2":"uy2":"cluster" with vectors\
	 head noborder size graph 0.03,20 lc var lw 2 title "mean(d+1)",\
	 "" using "x1":"y1":"ux1":"uy1":"cluster" with vectors\
	 head noborder size graph 0.03,20 lc var lw 2 title "mean(d-1)",\
	 "" using "x2":"y2":"cluster" with labels offset 1.2,1 font ",16" notitle
}
