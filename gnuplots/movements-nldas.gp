reset
set datafile separator ","
unset key

# set xtics 0.25 rotate format "%.2f"
# set ytics 0.25 format "%.2f"
# set grid
# set multiplot layout 

hucs="02070002"
ndays="1"
series="pds"


do for [i=1:1]{
    HUC=word(hucs, i)
    nday=word(ndays, i)
    set title sprintf("HUC: %s (%s day)", HUC, nday)
    
    files = system("ls -1 ../data/output/".HUC."/nldas/*.csv")
    plot "../data/output/".HUC."/basin.csv"u 1:2\
 	 with lines lw 2 lc "black" notitle,\
	 for [fil in files] fil u "y":"x":"dy":"dx":"cluster" with vectors lc var notitle
}

