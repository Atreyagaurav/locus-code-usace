reset
set datafile separator ","
unset key

set xtics 0.25 rotate format "%.2f"
set ytics 0.25 format "%.2f"
set grid
set multiplot layout 2,2

hucs="02070002"
ndays="1"
series="ams"


do for [i=1:1]{
    HUC=word(hucs, i)
    nday=word(ndays, i)
    set title "NLDAS Hourly"
    
    files = system("ls -1 ../data/output/".HUC."/nldas/*.csv")
    plot "../data/output/".HUC."/basin.csv"u 1:2\
 	 with lines lw 2 lc "black" notitle,\
	 for [fil in files] fil u "y":"x":"dy":"dx":"cluster" with vectors lc var notitle

    set title "NLDAS Daily Average"
    plot "../data/output/".HUC."/basin.csv"u 1:2\
 	 with lines lw 2 lc "black" notitle,\
	 "../data/output/".HUC."/nldas-daily.csv" u "y":"x":"dy":"dx":"cluster" with vectors lc var notitle

    set title "NLDAS Cluster Average"
    plot "../data/output/".HUC."/basin.csv"u 1:2\
 	 with lines lw 2 lc "black" notitle,\
	 "../data/output/".HUC."/nldas-mean.csv" u "y":"x":"dy":"dx":"cluster" with vectors lw 2 lc var notitle,\
	 "" using "y":"x":"cluster" with points ls 7\
	 lc var ps 1.6 title "mean(d)",\
	 "" using "y":"x":(sprintf("%d", column("cluster"))):"cluster"\
	 with labels offset 1.2,1 tc var font ",16" notitle

    set title "LivNeh Cluster Average"
    plot "../data/output/".HUC."/basin.csv"u ($1+360):2\
	 with lines lw 2 lc "black" notitle,\
	 "../data/output/".HUC."/".series."_".nday."dy_vectors_mean.csv"\
	 using "x2":"y2":"cluster" with points ls 7\
	 lc var ps 1.6 title "mean(d)",\
	 "" using "x2":"y2":"ux2":"uy2":"cluster" with vectors\
	 head noborder size graph 0.03,20 lc var lw 2 title "mean(d+1)",\
	 "" using "x1":"y1":"ux1":"uy1":"cluster" with vectors\
	 head noborder size graph 0.03,20 lc var lw 2 title "mean(d-1)",\
	 "" using "x2":"y2":"cluster":"cluster" with labels\
	 offset 1.2,1 tc var font ",16" notitle
}

