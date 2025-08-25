

# library
library(tidyverse)
library(stars)
library(rnaturalearth)
library(patchwork)

library(caret)
library(pdp)
library(vip)
library(iml)
library(purrr)
library(ggplot2)
library(reshape2)
library(corrplot)


rm(list=ls())


# processing data

df <- read.csv('data/TotalBaseDatos_ColombiaPeru_2025-04-25_v58.csv')
colnames(df)

df <- df %>% select(c(Climate, FarmUseCoffee,FarmArea, CoffeeLand,TypesShadeTrees, DistinctShadeTrees, 
                      production_farm_real, RH2M_avg, T2M, ph, temp_med,textura, pedrego, co, sat_alum, perd_sue, 
                      cic, sb, PR, production_farm_yolo))





# Select only numeric columns
numeric_data <- df[sapply(df, is.numeric)]

colnames(numeric_data)

colnames(numeric_data)[5] <- c("Manual Monitoring")
colnames(numeric_data)[14] <- c("Smarthphone Monitoring")




# Compute the Pearson correlation matrix
cor_matrix <- cor(numeric_data, method = "pearson", use = "complete.obs")

svg("results/Figure1_CorrelationVariables_2025-05-13_v01.svg", width = 8, height = 8) 

# Option 1: Corrplot (simple and popular)
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, 
         addCoef.col = "black", number.cex = 0.7)

dev.off()

