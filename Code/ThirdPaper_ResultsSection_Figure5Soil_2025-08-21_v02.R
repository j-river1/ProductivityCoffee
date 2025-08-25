 

# library
library(tidyverse)
library(stars)
library(rnaturalearth)
library(patchwork)
library(ggh4x)
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



r2.smarthphone  <- data.frame(r2 = c(r2.lm.smarthphone ,r2.cart.smarthphone ,r2.rf.smarthphone  ,r2.gbm.smarthphone ), algorithm = c("Linear model smarthphone","Decision Tree smarthphone","Random Forests smarthphone","Gradient Boosting smarthphone")) 
r2.manual  <- data.frame(r2 = c(r2.lm.manual ,r2.cart.manual ,r2.rf.manual ,r2.gbm.manual), algorithm = c("Linear model manual monitoring","Decision Tree manual monitoring","Random Forests manual monitoring","Gradient Boosting manual monitoring"))

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
                    pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "darkgreen", color="black")) + labs(title=paste0("Smartphone Monitoring \nRandom Forests ", paste("R² =", round(r2.rf.smarthphone, 3)))) +theme_bw()  + theme(
                      plot.title = element_text(size = 18, face = "bold"),   # Title size
                      axis.title.x = element_text(size = 16),                # X-axis label size
                      axis.title.y = element_text(size = 16),                # Y-axis label size
                      axis.text.x = element_text(size = 14),                 # X-axis text size
                      axis.text.y = element_text(size = 14)                  # Y-axis text size
                    )


pvip_rf.manual <- vip::vip(model.rf.manual, method="permute", train=data_train_manual, target="production_farm_real", metric="rsq_trad",
                                pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "darkgreen", color="black")) + labs(title=paste0("Manual Monitoring \nRandom Forests ", paste("R² =", round(r2.rf.manual, 3)))) +theme_bw()


set.seed(123)
pvip_gbm.smarthphone <- vip(model.gbm.smarthphone, method="permute", train=data_train_smarthphone, target="production_farm_yolo", metric="rsq_trad",
                pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "darkblue", color="black")) + labs(title=paste0("Smartphone Monitoring \nGradient Boosting ", paste("R² =", round(r2.gbm.smarthphone, 3)))) +theme_bw()


pvip_gbm.manual  <- vip(model.gbm.manual , method="permute", train=data_train_manual, target="production_farm_real", metric="rsq_trad",
                pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "darkblue", color="black")) + labs(title=paste0("Manual Monitoring \nGradient Boosting ", paste("R² =", round(r2.gbm.manual, 3)))) +theme_bw()


# aca no voy
#++++++++++


# Yield_CT * Tmax
pdp_FarmUseCoffee_CT   <- rbind(
  model.rf.smarthphone %>%  partial(pred.var=c("textura", "FarmUseCoffee"), approx=T) %>% cbind(., Monitoring = "Smartphone monitoring"),
  model.rf.manual %>%  partial(pred.var=c("textura", "FarmUseCoffee"), approx=T)  %>% cbind(., Monitoring = "Manual monitoring")
) 

pdp_FarmUseCoffee_CT$Monitoring <- factor(pdp_FarmUseCoffee_CT$Monitoring, levels=c("Smartphone monitoring","Manual monitoring"))

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

# Apply the translation to the 'textura' column
pdp_FarmUseCoffee_CT$Texture <- translation_map[pdp_FarmUseCoffee_CT$textura]




pdp_FarmUseCoffee_CT <- 
  pdp_FarmUseCoffee_CT  %>% 
  mutate(textura = case_when(
    Texture == "Sandy clay\nloam" ~ "Sandy clay loam",
    Texture != "Sandy clay\nloam" ~ "Other"
  ))



Fig06 <-
  ggplot(pdp_FarmUseCoffee_CT, aes(x=FarmUseCoffee, y=yhat, 
                            linetype=textura, color=textura)) +
  facet_wrap(vars(Monitoring)) +
  geom_line(stat="summary", fun=mean, size=1) +
  geom_vline(xintercept = 12.9, color = "red", linetype = "dashed", size = 1) + 
  ylab("Partial dependence") +
  theme_bw()  +
  scale_color_manual(values=c("Other"="darkgreen",
                              "Sandy clay loam"="blue")) +
  theme(
    axis.title = element_text(size = 16),      # Axis titles
    axis.text = element_text(size = 14),       # Axis tick labels
    legend.title = element_text(size = 14),    # Legend title
    legend.text = element_text(size = 12),     # Legend labels
    strip.text = element_text(size = 14)       # Facet labels
  ) + xlab("Farm Use for coffee (ha)") 

ggsave("results/Figure06_partialdepencePlotUmbral_2025-08-21_v03.svg", plot = Fig06, 
       width = 7, height = 4, dpi = 300, units = "in")













