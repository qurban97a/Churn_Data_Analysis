library(tidyverse)
library(skimr)
library(inspectdf)
library(caret)
library(glue)
library(highcharter)
library(h2o)
library(scorecard)
library(collapse)

# ----------------------------- Data Preprocessing -----------------------------

raw <- read.csv("Churn_Modelling (1).csv")

raw %>% view

names(raw)
raw %>% glimpse()

fNdistinct(raw)

raw %>% skim()

raw$Exited  %>% table() %>% prop.table() %>% round(2)


raw %>% inspect_na()

# Bizde burda $HasCrCard, $IsActiveMember sutunlari eslinde labelli sutunlardi deye bunlari once ceviririk factora
df <- raw %>% select(-RowNumber,-CustomerId,-Surname)
df$Exited <- df$Exited %>% as.factor()
df$HasCrCard <- df$HasCrCard %>% as.factor()
df$IsActiveMember <- df$IsActiveMember %>% as.factor()

df.num <- df %>% select_if(is.numeric)

df.chr <- df %>%
  mutate_if(is.character,as.factor) %>% 
  select_if(is.factor) %>% 
  select(Exited,everything())

#Changing outliers
num_vars <- df.num %>% names()

for_vars <- c()
for (b in 1:length(num_vars)) {
  OutVals <- boxplot(df.num[[num_vars[b]]], plot=T)$out
  if(length(OutVals)>0){
    for_vars[b] <- num_vars[b]
  }
}
for_vars <- for_vars %>% as.data.frame() %>% drop_na() %>% pull(.) %>% as.character()
for_vars %>% length()

for (o in for_vars) {
  OutVals <- boxplot(df.num[[o]])$out
  mean <- mean(df.num[[o]],na.rm=T)
  
  o3 <- ifelse(OutVals>mean,OutVals,NA) %>% na.omit() %>% as.matrix() %>% .[,1]
  o1 <- ifelse(OutVals<mean,OutVals,NA) %>% na.omit() %>% as.matrix() %>% .[,1]
  
  val3 <- quantile(df.num[[o]],0.75,na.rm = T) + 1.5*IQR(df.num[[o]],na.rm = T)
  df.num[which(df.num[[o]] %in% o3),o] <- val3
  
  val1 <- quantile(df.num[[o]],0.25,na.rm = T) - 1.5*IQR(df.num[[o]],na.rm = T)
  df.num[which(df.num[[o]] %in% o1),o] <- val1
}



# One Hote Encoding

ohe <- dummyVars(" ~ .", data = df.chr[,-1]) %>% 
  predict(newdata = df.chr[,-1]) %>% 
  as.data.frame()
ohe %>% glimpse()
df <- cbind(df.chr[1],ohe,df.num) 

ohe %>% view()

df %>% glimpse()


names(df)


# --------------------------------- Modeling ---------------------------------

# Weight Of Evidence ----

# IV (information values) 
iv <- df %>% 
  iv(y = 'Exited') %>% as_tibble() %>%
  mutate(info_value = round(info_value, 3)) %>%
  arrange(desc(info_value))


# Exclude not important variables 
ivars <- iv %>% 
  filter(info_value>0.02) %>% 
  select(variable) %>% .[[1]] 

df.iv <- df %>% select(Exited,ivars)

df.iv %>% dim()

# woe binning 
bins <- df.iv %>% woebin("Exited")

# breaking data into train and test & converting into woe values
dt_list <- df.iv %>%
  split_df("Exited", ratio = 0.8, seed = 123)

train_woe <- dt_list$train %>% woebin_ply(bins) 
test_woe <- dt_list$test %>% woebin_ply(bins)

names <- train_woe %>% names() %>% gsub("_woe","",.)                   
names(train_woe) <- names
names(test_woe) <- names
train_woe %>% inspect_na()
test_woe %>% inspect_na()


# Multicollinearity ----

# coef_na
target <- 'Exited'
features <- train_woe %>% select(-Exited) %>% names()

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = train_woe, family = "binomial")
glm %>% summary()

coef_na <- attributes(alias(glm)$Complete)$dimnames[[1]]
features <- features[!features %in% coef_na]
f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = train_woe, family = "binomial")

# VIF (Variance Inflation Factor) 
f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = train_woe, family = "binomial")

while(glm %>% vif() %>% arrange(desc(gvif)) %>% .[1,2] >= 1.5){
  afterVIF <- glm %>% vif() %>% arrange(desc(gvif)) %>% .[-1,] %>% pull(variable)
  f <- as.formula(paste(target, paste(afterVIF, collapse = " + "), sep = " ~ "))
  glm <- glm(f, data = train_woe, family = "binomial")
}

glm %>% vif() %>% arrange(desc(gvif)) %>% pull(variable) -> features 

features %>% view()


# Modeling with GLM ----

h2o.init()

train_h2o <- train_woe %>% select(target,features) %>% as.h2o()
test_h2o <- test_woe %>% select(target,features) %>% as.h2o()

model <- h2o.glm(
  x = features, y = target, family = "binomial", 
  training_frame = train_h2o, validation_frame = test_h2o,
  nfolds = 10, seed = 123, remove_collinear_columns = T,
  balance_classes = T, lambda = 0, compute_p_values = T)

while(model@model$coefficients_table %>%
      as.data.frame() %>%
      select(names,p_value) %>%
      mutate(p_value = round(p_value,3)) %>%
      .[-1,] %>%
      arrange(desc(p_value)) %>%
      .[1,2] >= 0.05){
  model@model$coefficients_table %>%
    as.data.frame() %>%
    select(names,p_value) %>%
    mutate(p_value = round(p_value,3)) %>%
    filter(!is.nan(p_value)) %>%
    .[-1,] %>%
    arrange(desc(p_value)) %>%
    .[1,1] -> v
  features <- features[features!=v]
  
  train_h2o <- train_woe %>% select(target,features) %>% as.h2o()
  test_h2o <- test_woe %>% select(target,features) %>% as.h2o()
  
  model <- h2o.glm(
    x = features, y = target, family = "binomial", 
    training_frame = train_h2o, validation_frame = test_h2o,
    nfolds = 10, seed = 123, remove_collinear_columns = T,
    balance_classes = T, lambda = 0, compute_p_values = T)
}
model@model$coefficients_table %>%
  as.data.frame() %>%
  select(names,p_value) %>%
  mutate(p_value = round(p_value,3))

model@model$coefficients %>%
  as.data.frame() %>%
  mutate(names = rownames(model@model$coefficients %>% as.data.frame())) %>%
  `colnames<-`(c('coefficients','names')) %>%
  select(names,coefficients)

h2o.varimp(model) %>% as.data.frame() %>% .[.$percentage != 0,] %>%
  select(variable, percentage) %>%
  hchart("pie", hcaes(x = variable, y = percentage)) %>%
  hc_colors(colors = 'orange') %>%
  hc_xAxis(visible=T) %>%
  hc_yAxis(visible=T)


# ---------------------------- Evaluation Metrices ----------------------------

# Prediction & Confision Matrice
pred <- model %>% h2o.predict(newdata = test_h2o) %>% 
  as.data.frame() %>% select(p1,predict)

model %>% h2o.confusionMatrix(newdata = test_h2o)

model %>% h2o.performance(newdata = test_h2o) %>%
  h2o.find_threshold_by_max_metric('f1')

eva <- perf_eva(
  pred = pred %>% pull(p1),
  label = dt_list$test$Exited %>% as.character() %>% as.numeric(),
  binomial_metric = c("auc","gini"),
  show_plot = "roc")

eva$binomial_metric$dat

model %>%
  h2o.auc(train = T,
          valid = T,
          xval = T) %>%
  as_tibble() %>%
  round(2) %>%
  mutate(data = c('train','test','cross_val')) %>%
  mutate(gini = 2*value-1) %>%
  select(data,auc=value,gini)