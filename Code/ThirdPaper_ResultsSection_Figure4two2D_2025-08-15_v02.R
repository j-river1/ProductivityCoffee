

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

rm(list=ls())


# processing data

df <- read.csv('data/TotalBaseDatos_ColombiaPeru_2025-04-25_v58.csv')
colnames(df)



# df <- df %>% dplyr::select(-c(Latitude, Longitude, Altitude, RealCherriesBranch))
# ph :La acidez (pH) es el logaritmo negativo de la actividad de iones H+ en la solución o suspensión del suelo
# perfil: Perfíl modal del suelo
# pendiente:Es la inclinación de un terreno respecto a un plano horizontal que pasa por su base
# Pedregocidad: pedrego
# humedad: Humedad del suelo
# temp_med : Estado térmico del aire Medida del estado térmico del aire
# ce: Coducción de corriente eléctica
# p_textura: 	Puntos asignados en la tabla de puntuación general para la variable de textura
# p_temp: Puntos asignados en la tabla de puntuación general para la variable de temperatura
# p_profundi: Puntos asignados en la tabla de puntuación general para la variable de profundidad
# p_ph: Puntos asignados en la tabla de puntuación general para la variable de ph
# p_pend: Puntos asignados en la tabla de puntuación general para la variable de pendiente 
# p_pedrego: Puntos asignados en la tabla de puntuación general para la variable de pedregocidad
# p_co: Puntos asignados en la tabla de puntuación general para la variable de monóxido de Carbono
# p_sat_alum: Puntos asignados en la tabla de puntuación general para la variable de la saturación de aluminio
# p_salini: 	Puntos asignados en la tabla de puntuación general para la variable de la sanidad
# p_sodio:  Puntos asignados en la tabla de puntuación general para la variable de la presencia de sodio
# p_perd_sue: Puntos asignados en la tabla de puntuación general para la variable de pérdida de suelo
# p_cic: Puntos asignados en la tabla de puntuación general para la variable de la capacidad de intercambio catiónico
# p_inund: Puntos asignados en la tabla de puntuación general para la variable de inundación
# p_sb Puntos asignados en la tabla de puntuación general para la variable de intercambio catiónico
# p_humedad: Puntos asignados en la tabla de puntuación general para la variable de humedad
# p_drenaje: Puntos asignados en la tabla de puntuación general para la variable de drenaje natural
# cic: Capacidad de intercambio catiónico
#  etp_ppt: 	 Índice de la relación mensual entre la evapotranspiración actual


df <- df %>% select(c(Climate, FarmUseCoffee,FarmArea, X_GPS_altitude, CoffeeLand,TypesShadeTrees, DistinctShadeTrees, 
                      production_farm_real, RH2M_avg, T2M, ph, temp_med ,
                      ce,textura, pedrego, co, sat_alum, perd_sue, 
                      cic, sb, X_GPS_latitude, X_GPS_longitude, X_GPS_altitude, PR, production_farm_yolo))

df_smarthphone <- df %>% dplyr::select(-c(X_GPS_latitude, X_GPS_longitude, X_GPS_altitude, production_farm_real))

df_manual <- df %>% dplyr::select(-c(X_GPS_latitude, X_GPS_longitude, X_GPS_altitude, production_farm_yolo))


df_smarthphone <- na.omit(df_smarthphone)
df_manual <- na.omit(df_manual)


# Split data for machine learning -----------------------------------------

set.seed(224)
train_test_split_df_smarthphone <- sample(c(1:nrow(df_smarthphone)), 0.8*nrow(df_smarthphone), replace=F) 
data_train_smarthphone <- df_smarthphone[train_test_split_df_smarthphone,]
data_test_smarthphone  <- setdiff(df_smarthphone , data_train_smarthphone)


train_test_split_df_manual <- sample(c(1:nrow(df_manual)), 0.8*nrow(df_manual), replace=F) 
data_train_manual <- df_manual[train_test_split_df_manual,]
data_test_manual  <- setdiff(df_manual , data_train_manual)





# Machine learning algorithm implementation -------------------------------


tc = trainControl(method = "cv", number = 5) # caret
tune_grid <- expand.grid(mincriterion = 0.50, maxdepth = 30)


set.seed(132)
model.lm.smarthphone  = caret::train(production_farm_yolo ~., data=data_train_smarthphone, method="glmStepAIC", trControl=tc)
model.lm.manual  = caret::train(production_farm_real ~., data=data_train_manual, method="glmStepAIC", trControl=tc)


set.seed(132)
model.cart.smarthphone = caret::train(production_farm_yolo ~., data=data_train_smarthphone, method="ctree2", trControl=tc,  tuneGrid = tune_grid)
model.cart.manual = caret::train(production_farm_real ~., data=data_train_manual, method="ctree2", trControl=tc,  tuneGrid = tune_grid)




set.seed(132)
model.rf.smarthphone   = caret::train(production_farm_yolo ~., data=data_train_smarthphone, method="rf", trControl=tc)
model.rf.manual   = caret::train(production_farm_real ~., data=data_train_manual, method="rf", trControl=tc)




set.seed(132)
model.gbm.smarthphone  = caret::train(production_farm_yolo ~., data=data_train_smarthphone, method="gbm", trControl=tc,
                                      tuneGrid=expand.grid(n.trees=(1:5)*500, interaction.depth=(1:5)*3,
                                                           shrinkage=0.1, n.minobsinnode=10))


model.gbm.manual  = caret::train(production_farm_real ~., data=data_train_manual, method="gbm", trControl=tc,
                                 tuneGrid=expand.grid(n.trees=(1:5)*500, interaction.depth=(1:5)*3,
                                                      shrinkage=0.1, n.minobsinnode=10))




# 2-way interactions


# Define the translation mapping
translation_map <- c(
  "Arcillosa" = "Clay",
  "Franca" = "Loam",
  "Franco arcillo arenosa" = "Sandy clay\nloam",
  "Franco arcillosa" = "Clay\nloam",
  "Franco arenosa" = "Sandy\nloam",
  "Franco limosa" = "Silty\nloam",
  "No aplica" = "Not\napplicable"
)

# Smartphone model PDP
pdp_rf_smartphone <- model.rf.smarthphone %>%
  partial(pred.var = c("FarmUseCoffee", "textura"), approx = TRUE) %>%
  mutate(textura = recode(textura, !!!translation_map)) %>% 
  autoplot +
  labs(title = "Smartphone Monitoring") + theme_bw() +
theme(
    plot.title   = element_text(size = 14, face = "bold"),
    axis.title   = element_text(size = 14),
    axis.text    = element_text(size = 14),
    legend.title = element_text(size = 14, face = "bold"),
    legend.text  = element_text(size = 14),
    strip.text   = element_text(size = 14, face = "bold") # facet labels
  )

# Manual model PDP
pdp_rf_manual <- model.rf.manual %>%
  partial(pred.var = c("FarmUseCoffee", "textura"), approx = TRUE) %>%
  mutate(textura = recode(textura, !!!translation_map)) %>% 
  autoplot +
  labs(title = "Manual Monitoring") + theme_bw() +
  theme(
    plot.title   = element_text(size = 14, face = "bold"),
    axis.title   = element_text(size = 14),
    axis.text    = element_text(size = 14),
    legend.title = element_text(size = 14, face = "bold"),
    legend.text  = element_text(size = 14),
    strip.text   = element_text(size = 14, face = "bold") # facet labels
  )



Fig07 <- pdp_rf_smartphone + pdp_rf_manual

ggsave(paste0("results/Figure4_2DSmartManual_2025-08-14_v01.svg"), plot = Fig07 , dpi = 300, width = 10, height = 6)


