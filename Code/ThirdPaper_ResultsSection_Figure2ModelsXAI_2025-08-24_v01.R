 

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




# -------------------------------------------------------
# performance evaluation

# with data_test
pred.lm.smarthphone   <- predict(model.lm.smarthphone , data_test_smarthphone)
pred.lm.manual   <- predict(model.lm.manual , data_test_manual)


#
pred.cart.smarthphone <- predict(model.cart.smarthphone, data_test_smarthphone)
pred.cart.manual <- predict(model.cart.manual, data_test_manual)




pred.rf.smarthphone <- predict(model.rf.smarthphone, data_test_smarthphone)
pred.rf.manual  <- predict(model.rf.manual, data_test_manual)


pred.gbm.smarthphone  <- predict(model.gbm.smarthphone, data_test_smarthphone)
pred.gbm.manual  <- predict(model.gbm.manual, data_test_manual)


# r2: obs vs pred
r2.lm.smarthphone   <- cor(pred.lm.smarthphone , data_test_smarthphone$production_farm_yolo)^2 %>% round(.,4)
r2.lm.manual   <- cor(pred.lm.manual , data_test_manual$production_farm_real)^2 %>% round(.,4)



r2.cart.smarthphone <- cor(pred.cart.smarthphone, data_test_smarthphone$production_farm_yolo)^2 %>% round(.,4)
r2.cart.manual  <- cor(pred.cart.manual , data_test_manual$production_farm_real)^2 %>% round(.,4)



r2.rf.smarthphone   <- cor(pred.rf.smarthphone, data_test_smarthphone$production_farm_yolo)^2 %>% round(.,4)
r2.rf.manual   <- cor(pred.rf.manual, data_test_manual$production_farm_real)^2 %>% round(.,4)


rmse.rf.smarthphone <- sqrt(mean((pred.rf.smarthphone - data_test_smarthphone$production_farm_yolo)^2))
rmse.rf.manual <- sqrt(mean((pred.rf.manual - data_test_manual$production_farm_real)^2))


r2.gbm.smarthphone  <- cor(pred.gbm.smarthphone, data_test_smarthphone$production_farm_yolo)^2 %>% round(.,4)
r2.gbm.manual  <- cor(pred.gbm.manual, data_test_manual$production_farm_real)^2 %>% round(.,4)



r2.smarthphone  <- data.frame(r2 = c(r2.lm.smarthphone ,r2.cart.smarthphone ,r2.rf.smarthphone  ,r2.gbm.smarthphone ), algorithm = c("Linear model smarthphone","Decision Tree smarthphone","Random Forest smarthphone","Gradient Boosting smarthphone")) 
r2.manual  <- data.frame(r2 = c(r2.lm.manual ,r2.cart.manual ,r2.rf.manual ,r2.gbm.manual), algorithm = c("Linear model manual monitoring","Decision Tree manual monitoring","Random Forest manual monitoring","Gradient Boosting manual monitoring"))

r2.total <- rbind(r2.smarthphone, r2.manual )

r2.total


#-----------------------------------------------------------
# Interpretable machine learning

# variable importance (feature importance)

# model agnostic post-hoc interpretability
# permutation-based feature importance
set.seed(123)

pvip_lm.smarthphone <- vip(model.lm.smarthphone , method="permute", train=data_train_smarthphone, target="production_farm_yolo", metric="rsq_trad",
               pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "grey", color="black")) + labs(title="Linear model") +theme_bw() 
  


pvip_lm.manual <- vip(model.lm.manual , method="permute", train=data_train_manual, target="production_farm_real", metric="rsq_trad",
               pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "grey", color="black")) + labs(title="Linear model") +theme_bw()



set.seed(123)

pvip_cart.smarthphone <- vip(model.cart.smarthphone, method="permute", train=data_train_smarthphone, target="production_farm_yolo", metric="rsq_trad",
                 pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "orange", color="black")) + labs(title="Decision Tree") +theme_bw() +
                 annotate("text", x = Inf, y = -Inf, hjust = 1.1, vjust = -1.2, label = paste("R² =", round(r2.cart.smarthphone, 3)), size = 5, color = "black") 


pvip_cart.manual <- vip(model.cart.manual, method="permute", train=data_train_manual, target="production_farm_real", metric="rsq_trad",
                 pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "orange", color="black")) + labs(title="Decision Tree") +theme_bw() 



set.seed(123)

pvip_rf.smarthphone <- vip::vip(model.rf.smarthphone, method="permute", train=data_train_smarthphone, target="production_farm_yolo", metric="rsq_trad",
                    pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "darkgreen", color="black")) + labs(title=paste0("Smartphone Monitoring \nRandom Forest ", paste("R² =", round(r2.rf.smarthphone, 3)))) +theme_bw()  + 
                      theme(
                        plot.title = element_text(size = 14, face = "bold"),   # Title size
                        axis.title.x = element_text(size = 14),                # X-axis label size
                        axis.title.y = element_text(size = 14),                # Y-axis label size
                        axis.text.x = element_text(size = 14),                 # X-axis text size
                        axis.text.y = element_text(size = 14)                  # Y-axis text size
                      )
                      

pvip_rf.manual <- vip::vip(model.rf.manual, method="permute", train=data_train_manual, target="production_farm_real", metric="rsq_trad",
                                pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "darkgreen", color="black")) + labs(title=paste0("Manual Monitoring \nRandom Forest ", paste("R² =", round(r2.rf.manual, 3)))) +theme_bw() + 
theme(
  plot.title = element_text(size = 14, face = "bold"),   # Title size
  axis.title.x = element_text(size = 14),                # X-axis label size
  axis.title.y = element_text(size = 14),                # Y-axis label size
  axis.text.x = element_text(size = 14),                 # X-axis text size
  axis.text.y = element_text(size = 14)                  # Y-axis text size
)  

set.seed(123)
pvip_gbm.smarthphone <- vip(model.gbm.smarthphone, method="permute", train=data_train_smarthphone, target="production_farm_yolo", metric="rsq_trad",
                pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "darkblue", color="black")) + labs(title=paste0("Smartphone Monitoring \nGradient Boosting ", paste("R² =", round(r2.gbm.smarthphone, 3)))) +theme_bw()


pvip_gbm.manual  <- vip(model.gbm.manual , method="permute", train=data_train_manual, target="production_farm_real", metric="rsq_trad",
                pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "darkblue", color="black")) + labs(title=paste0("Manual Monitoring \nGradient Boosting ", paste("R² =", round(r2.gbm.manual, 3)))) +theme_bw()




# patchwork



# plot_pvip_all <-  pvip_rf.smarthphone  + pvip_rf.manual + pvip_gbm.smarthphone + pvip_gbm.manual

plot_pvip_all <-  pvip_rf.smarthphone  + pvip_rf.manual 



Fig03 <-
  plot_pvip_all+ 
  plot_annotation(tag_levels = 'a')

Fig03


ggsave(paste0("results/Figure2_permutation-basedFeatureImportanceYOLO_2025-08-24_v16.svg"), plot = Fig03 , dpi = 300, width = 10, height = 6)



# #-----------------------hasta aqui va ##############################################
# 
# # 
# # 
# # 
# # ggsave(paste0("results/Figure2_permutation-basedFeatureImportanceYOLO_2025-05-13_v13.svg"), plot = Fig03 , dpi = 300, width = 10, height = 6)
# # 
# 
# 
# # partial dependence plot FarmUseCoffee,FarmArea
# 
# 
# pdp_FarmUseCoffee_CT   <- rbind(
#   model.rf.smarthphone %>%  partial(pred.var=c("FarmUseCoffee"), approx=T) %>% cbind(., Monitoring = "RF Smarthphone"),
#   model.rf.manual %>%  partial(pred.var=c("FarmUseCoffee"), approx=T)  %>% cbind(., Monitoring = "RF Manual")
# ) 
# pdp_FarmUseCoffee_CT$Monitoring <- factor(pdp_FarmUseCoffee_CT$Monitoring, levels=c("RF Smarthphone","RF Manual"))
# 
# 
# pdp_FarmArea_CT    <- rbind(
#   model.rf.smarthphone %>%  partial(pred.var=c("FarmArea"), approx=T) %>% cbind(., Monitoring = "RF Smarthphone"),
#   model.rf.manual %>%  partial(pred.var=c("FarmArea"), approx=T)  %>% cbind(., Monitoring = "RF Manual")
# )
# pdp_FarmArea_CT$Monitoring <- factor(pdp_FarmArea_CT$Monitoring, levels=c("RF Smarthphone","RF Manual"))
# 
# 
# Fig05a <-
#   ggplot(pdp_FarmUseCoffee_CT , aes(x=FarmUseCoffee, y=yhat, color= Monitoring)) +
#   geom_line(size=1) +
#   scale_color_manual(values=c("RF Smarthphone"="darkgreen",
#                               "RF Manual"="darkblue"))  +
#   ylab("Partial dependence") + xlab("")+
#   theme_bw() +   theme(legend.position=c(0.3,0.9),
#                        axis.text.x = element_blank(), 
#                        axis.text.y = element_text(size = 14),
#                        axis.title.y = element_text(size = 14),
#                        legend.title = element_text(size = 14),
#                        legend.text = element_text(size = 14))
# 
# data_train_smarthphone$Monitoring <- "Smartphone"
# data_train_manual$Monitoring  <- "Manual"
# 
# data_train_manual_ <- data_train_manual %>% select(!(production_farm_real))
# data_train_smarthphone_ <- data_train_smarthphone %>% select(!(production_farm_yolo))
# 
# 
# data_combined <- rbind(data_train_smarthphone_, data_train_manual_)
# 
# Fig05b <-ggplot(data_combined, aes(x = FarmUseCoffee, fill = Monitoring)) +
#   geom_histogram(position = "identity", alpha = 0.5, bins = 50) +
#   scale_fill_manual(values = c(
#     "Smartphone" = "darkgreen",
#     "Manual" = "darkblue"
#   )) +
#   theme_bw() +
#   labs(x = "Farm Use for Coffee (ha)", y = "Count", fill = "Monitoring") +theme(legend.position="none",
#                                                                                 axis.text.y = element_text(size = 12),
#                                                                                 axis.text.x = element_text(size = 12),
#                                                                                 axis.title.y = element_text(size = 14),
#                                                                                 axis.title.x = element_text(size = 14))
# 
# Fig05c <-
#   ggplot(pdp_FarmArea_CT , aes(x=FarmArea, y=yhat, color= Monitoring)) +
#   geom_line(size=1) +
#   scale_color_manual(values=c("RF Smarthphone"="darkgreen",
#                               "RF Manual"="darkblue"))  +
#   ylab("Partial dependence") + xlab("")+
#   theme_bw()+ theme(legend.position="none", 
#                     axis.title.x = element_blank(),
#                     axis.text.x  = element_blank(),
#                     axis.text.y = element_text(size = 14),
#                     axis.title.y = element_text(size = 14),
#                     legend.title = element_text(size = 14),
#                     legend.text = element_text(size = 14))
# 
# 
# Fig05d <-ggplot(data_combined, aes(x = FarmArea, fill = Monitoring)) +
#   geom_histogram(position = "identity", alpha = 0.5, bins = 50) +
#   scale_fill_manual(values = c(
#     "Smartphone" = "darkgreen",
#     "Manual" = "darkblue"
#   )) +
#   theme_bw() +
#   labs(x = "Farm Area (ha)", y = "Count", fill = "Monitoring") +theme(legend.position="none", 
#                                                                       axis.text.y = element_text(size = 12),
#                                                                       axis.text.x = element_text(size = 12),
#                                                                       axis.title.y = element_text(size = 14),
#                                                                       axis.title.x = element_text(size = 14))
# 
# 
# 
# 
# Fig05 <- Fig05a + Fig05c + Fig05b + Fig05d +
#   plot_annotation(tag_levels = "a") +
#   plot_layout(heights=c(9,1))
# 
# # Fig05
# # 
# # ggsave(paste0("results/Figure3_partialdepencePlot_2025-05-16_v01.svg"), plot = Fig05 , dpi = 300, width = 10, height = 6)
# 
# 
# 
# 
# # partial dependence plot FarmUseCoffee, Textura 
# 
# 
# pdp_FarmUseCoffee_CT   <- rbind(
#   model.rf.smarthphone %>%  partial(pred.var=c("FarmUseCoffee"), approx=T) %>% cbind(., Monitoring = "RF Smarthphone"),
#   model.rf.manual %>%  partial(pred.var=c("FarmUseCoffee"), approx=T)  %>% cbind(., Monitoring = "RF Manual")
# ) 
# pdp_FarmUseCoffee_CT$Monitoring <- factor(pdp_FarmUseCoffee_CT$Monitoring, levels=c("RF Smarthphone","RF Manual"))
# 
# 
# pdp_textura_CT    <- rbind(
#   model.rf.smarthphone %>%  partial(pred.var=c("textura"), approx=T) %>% cbind(., Monitoring = "RF Smarthphone"),
#   model.rf.manual %>%  partial(pred.var=c("textura"), approx=T)  %>% cbind(., Monitoring = "RF Manual")
# )
# pdp_textura_CT$Monitoring <- factor(pdp_textura_CT$Monitoring, levels=c("RF Smarthphone","RF Manual"))
# 
# 
# Fig05a <-
#   ggplot(pdp_FarmUseCoffee_CT , aes(x=FarmUseCoffee, y=yhat, color= Monitoring)) +
#   geom_line(size=1) +
#   scale_color_manual(values=c("RF Smarthphone"="darkgreen",
#                               "RF Manual"="darkblue"))  +
#   ylab("Partial dependence") + xlab("")+
#   theme_bw() +   theme(legend.position=c(0.3,0.9), axis.title.x = element_blank(),
#                        axis.text.x  = element_blank(),
#                        axis.text.y = element_text(size = 12),
#                        axis.title.y = element_text(size = 12))
# 
# 
# 
# data_train_smarthphone$Monitoring <- "Smartphone"
# data_train_manual$Monitoring  <- "Manual"
# 
# data_train_manual_ <- data_train_manual %>% select(!(production_farm_real))
# data_train_smarthphone_ <- data_train_smarthphone %>% select(!(production_farm_yolo))
# 
# 
# data_combined <- rbind(data_train_smarthphone_, data_train_manual_)
# 
# Fig05b <-ggplot(data_combined, aes(x = FarmUseCoffee, fill = Monitoring)) +
#   geom_histogram(position = "identity", alpha = 0.5, bins = 50) +
#   scale_fill_manual(values = c(
#     "Smartphone" = "darkgreen",
#     "Manual" = "darkblue"
#   )) +
#   theme_bw() +
#   labs(x = "Farm Use for Coffee (ha)", y = "Count", fill = "Monitoring") +theme(legend.position="none")
# 
# Fig05c <-
#   ggplot(pdp_textura_CT  , aes(x= textura, y=yhat, color= Monitoring)) +
#   geom_line(size=1) +
#   scale_color_manual(values=c("RF Smarthphone"="darkgreen",
#                               "RF Manual"="darkblue"))  +
#   ylab("Partial dependence") + xlab("")+
#   theme_bw()+ theme(legend.position="none", axis.title.x = element_blank(),
#                     axis.text.x  = element_blank())
# 
# 
# # Define the translation mapping
# translation_map <- c(
#   "Arcillosa" = "Clay",
#   "Franca" = "Loam",
#   "Franco arcillo arenosa" = "Sandy clay\nloam",
#   "Franco arcillosa" = "Clay\nloam",
#   "Franco arenosa" = "Sandy\nloam",
#   "Franco limosa" = "Silty\nloam",
#   "No aplica" = "Not\napplicable"
# )
# 
# # Apply the translation to the 'textura' column
# pdp_textura_CT$Texture <- translation_map[pdp_textura_CT$textura]
# 
# 
# 
# Fig05c <- ggplot(pdp_textura_CT, aes(x = Texture, y = yhat, fill = Monitoring)) +
#   geom_col(position = position_dodge(width = 0.8), width = 0.7) +
#   scale_fill_manual(values = c("RF Smarthphone" = "darkgreen", "RF Manual" = "darkblue")) +
#   ylab("Partial dependence") +
#   xlab("Soil Texture") +
#   theme_bw() + theme(legend.position="none", axis.title.x = element_blank(),
#                      axis.text.x  = element_blank(), 
#                      axis.text.y = element_text(size = 12),
#                      axis.title.y = element_text(size = 12))
# Fig05c
# 
# data_combined$Texture <- translation_map[data_combined$textura]
# 
# Fig05d <- ggplot(data_combined, aes(x = Texture)) +
#   geom_bar(fill = "steelblue") +
#   labs(x = "Soil Texture", y = "Count") +
#   theme_minimal()
# 
# 
# Fig05 <- Fig05a + Fig05c + Fig05b + Fig05d +
#   plot_annotation(tag_levels = "a") +
#   plot_layout(heights=c(9,1))
# 
# Fig05
# # 
# # ggsave(paste0("results/Figure3_partialdepencePlotSoilArea_2025-05-20_v03.svg"), plot = Fig05 , dpi = 300, width = 10, height = 6)
# # 
# # 
# 
# 
# 
# 
# 
# 
# 
# 
