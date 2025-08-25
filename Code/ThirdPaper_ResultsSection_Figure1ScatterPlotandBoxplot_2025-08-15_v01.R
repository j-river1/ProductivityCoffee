

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
library(dplyr)

rm(list=ls())


df <- read.csv('data/TotalBaseDatos_ColombiaPeru_2025-04-25_v58.csv')
colnames(df)



# Results section
# 3.1 Stadisitical analisys. 


df_aux <- df %>% select (production_farm_yolo, production_farm_real)

head(df_aux )

# Calculate summary statistics
summary_stats <- data.frame(
  Metric = c("Mean", "SD", "Median", "Min", "Max"),
  production_farm_yolo = c(mean(df_aux$production_farm_yolo), 
                           sd(df_aux$production_farm_yolo),
                           median(df_aux$production_farm_yolo),
                           min(df_aux$production_farm_yolo),
                           max(df_aux$production_farm_yolo)),
  production_farm_real = c(mean(df_aux$production_farm_real), 
                           sd(df_aux$production_farm_real),
                           median(df_aux$production_farm_real),
                           min(df_aux$production_farm_real),
                           max(df_aux$production_farm_real))
)

# Print summary statistics
summary_stats[ , 2:3] <- round(summary_stats[ , 2:3], 2)
summary_stats

df_aux <- na.omit(df_aux)


# Fit linear model
model <- lm(production_farm_real ~ production_farm_yolo, data = df_aux)

# Get R-squared
r2 <- summary(model)$r.squared

# Compute RMSE
rmse <- sqrt(mean((df_aux$production_farm_real - df_aux$production_farm_yolo)^2))
print(paste("RMSE:", round(rmse, 2)))




axis_limits <- range(c(df_aux$production_farm_yolo, df_aux$production_farm_real), na.rm = TRUE)


# Plot with ggplot2
plt <- ggplot(df_aux, aes(x = production_farm_yolo, y = production_farm_real)) +
  geom_point(color = "darkblue", alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40", size = 0.5) + 
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "",
       x = "Smartphone Monitoring \n[Cherries per tree]",
       y = "Manual Monitoring \n[Cherries per tree]") +
  annotate("text", x = Inf, y = -Inf, hjust = 1.1, vjust = -1.2,
           label = paste("R² =", round(r2, 3)),
           size = 5, color = "black") +
  theme_bw() + 
  coord_fixed() +
  theme(
    axis.text = element_text(size = 16),
    axis.title.x = element_text(size = 16),
    axis.text.x = element_text(size = 12),
    axis.title.y = element_text(size = 16),
  )+
  scale_x_continuous(limits = axis_limits, breaks = pretty(axis_limits)) +
  scale_y_continuous(limits = axis_limits, breaks = pretty(axis_limits)) 



df_long <- df_aux %>%
  pivot_longer(
    cols = c(production_farm_yolo, production_farm_real),
    names_to = "source",
    values_to = "cherries"
  ) %>%
  dplyr::mutate(source = factor(source,
                                levels = c("production_farm_yolo", "production_farm_real"),
                                labels = c("Smartphone \nMonitoring", "Manual \nMonitoring")))


plt_box <- ggplot(df_long, aes(x = source, y = cherries, fill = source)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    x = "",
    y = "",
  ) +
  theme_bw() +
  scale_fill_manual(values = c("Smartphone \nMonitoring" = "steelblue", 
                               "Manual \nMonitoring" = "forestgreen"))+
  theme(
    axis.text = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    plot.title = element_text(size = 16, face = "bold"),
    legend.position = "none"
  )

combined_plot <- plt + plt_box + plot_layout(ncol = 2)


ggsave("results/Figure1_ScatterplotBoxplotYOLOvsReal_21-08-2025_v6.svg", plot = combined_plot, 
       width = 7, height = 4, dpi = 300, units = "in")




