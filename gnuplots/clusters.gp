HUC="1203"

set datafile separator ","

set boxwidth 0.4
set style fill solid
set yrange [0:]
set y2range [0:]
set ylabel "Count"
set ytics nomirror
set y2tics
set y2label "Basin Averaged Precipitation (mm)"


plot "data/output/".HUC."/clusters-summary-ams_1day.csv"\
     using ($0-.2):"count":xtic(1) with boxes title "Count",\
     "" using ($0+.2):"precip" with boxes title "Basin Precip" axes x1y2
