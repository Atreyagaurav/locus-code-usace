from src.huc import HUC
import src.cluster as cluster
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearson3

h = HUC("1203")
h = HUC("02070002")
ams = cluster.clustered_df(h, "ams", 1)
ams.loc[:, "series"] = "ams"
# pds = cluster.clustered_df(h, "pds", 1)
# pds.loc[:, "series"] = "pds"
# df = pd.concat([ams, pds])
df = ams
df = df.set_index("end_date")
df.index.name = "time"
precip = h.load_timeseries()
data = df.join(precip.prec)
data.loc[:, "cluster"] = data.cluster.map(lambda s: s[2:]).astype(int)
sns.boxplot(data=data, x="cluster", y="prec", hue="series")

plt.show()


ams = ams.set_index("end_date")
ams.index.name = "time"
data = ams.join(precip.prec)
params = pearson3.fit(data.prec)
print("all:",pearson3.ppf(0.99, *params))
for c, data in data.groupby("cluster"):
    params = pearson3.fit(data.prec)
    print(f"{c}:",pearson3.ppf(0.99, *params))
