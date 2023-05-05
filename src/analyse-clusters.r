library(ggplot2)
library(lubridate)
library(gridExtra)

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
