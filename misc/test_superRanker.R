library(tidyverse)
library(SuperRanker)

data = read_csv('y_test.csv')

y = as.matrix(data)
