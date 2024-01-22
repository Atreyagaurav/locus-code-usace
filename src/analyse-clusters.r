library(ggplot2)
library(lubridate)
library(extRemes)
library(gridExtra)
library(dplyr)
library(dataRetrieval)

precip <- read.csv("data/output/05/pds_1dy_series.csv")
clusters <- read.csv("data/output/05/clusters-pds_1day.csv")

precip$cluster = clusters$cluster
precip$date = as.Date(precip$end_date)
precip$julian = lubridate::yday(precip$date)
precip$month = lubridate::month(precip$date)

p1 <- ggplot(data=precip) + geom_point(mapping = aes(x = julian, y = p_mm, color = cluster))
p2 <- ggplot(data=precip) + geom_boxplot(mapping = aes(x = cluster, y = p_mm, color = cluster))
p3 <- ggplot(data=precip) + geom_bar(mapping = aes(y = month, fill = cluster)) + coord_flip()
p4 <- ggplot(data=precip) + geom_line(mapping = aes(x = date, y = p_mm)) + geom_point(mapping = aes(x = date, y = p_mm, color = cluster))

grid.arrange(p1, p2, p3, p4, ncol=2)

ams <- read.csv("/tmp/precip.csv")

ams_pr <- fevd(x = ams$prec, method = "Lmoments", type="GEV", units = "cfs")
plot(ams_pr)

df1 <- readNWISuv("01603000", "00045", "2013-10-01", "2023-11-29")


df2 <- df1 %>% group_by(date=strftime(as.Date(df1$dateTime), "%Y-%m-%d")) %>% summarise(precip = sum(X_00045_00000)) %>% na.omit()
df3 <- df2 %>% group_by(year=strftime(as.Date(date), "%Y")) %>% summarise(precip = max(precip))

summary(df3$precip)

ams_pr <- fevd(x = df3$precip, method = "Lmoments", type="GEV", units = "cfs")
plot(ams_pr)
