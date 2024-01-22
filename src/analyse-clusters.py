import pandas as pd
import matplotlib.pyplot as plt

precip = pd.read_csv("data/output/05/pds_1dy_series.csv")
clusters = pd.read_csv("data/output/05/clusters-pds_1day.csv")

precip.loc[:, "cluster"] = clusters.cluster
precip.loc[:, "julian"] = [d.dayofyear for d in pd.DatetimeIndex(precip.end_date)]
precip.loc[:, "month"] = [d.month for d in pd.DatetimeIndex(precip.end_date)]

monthly_counts = precip.groupby(["month", "cluster"]).cluster.count()
monthly_counts.name = "counts"

cluster_max = monthly_counts.unstack("cluster").max().cumsum()

clus = sorted(clusters.cluster.unique())
bottoms = {c: cluster_max.loc[c] for c in clus}
colors = {c:c1 for c,c1 in zip(clus,plt.colormaps["Set2_r"].colors)}

for (month, cluster), val in monthly_counts.items():
    plt.bar(month, val, bottom=bottoms[cluster] - val / 2, color=colors[cluster])

plt.show()
