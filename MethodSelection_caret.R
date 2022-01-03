# libraries used
library(FNN)
library(glmnet)
library(caret)

############# Data Prep #############
bodyfat = read.csv("bodyfat.csv")

dim(bodyfat)
n = dim(bodyfat)[1]
names(bodyfat)  # using BodyFatSiri as response, versus Density or BodyFatBrozek

############# Model specifications #############
# Linear
Model1 = (BodyFatSiri ~ Weight+BMI+Height) #dimensions
Model2 = (BodyFatSiri ~ . - Case - BodyFatBrozek - Density) # all
allLinModels = list(Model1,Model2)	
# LASSO
#alllambda = exp(-100:20/10)
alllambda = exp(-100:15/10)
# kNN 
allkNN = 1:20

############# Cross-validation via caret package #############
# apply functions to get RMSE for linear Model1, linear Model2, best kNN, and best LASSO

#specify data to be used
dataused=bodyfat

# set up training method
set.seed(81)
training = trainControl(method = "cv", number = 10)

# cross-validation of linear model 1
fit_caret_lm1 = train(BodyFatSiri ~ Weight+BMI+Height,
                      data = dataused,
                      method = "lm",
                      trControl = training)
  # output
fit_caret_lm1

# cross-validation of linear model 2
fit_caret_lm2 = train(BodyFatSiri ~ . - Case - BodyFatBrozek - Density,
                      data = dataused,
                      method = "lm",
                      trControl = training)
  # output
fit_caret_lm2

# cross-validation of LASSO
fit_caret_LASSO = train(BodyFatSiri ~ . - Case - BodyFatBrozek - Density,
                      data = dataused,
                      method = "glmnet",
                      trControl = training,
                      tuneGrid = expand.grid(alpha=c(1),lambda=alllambda))
# output and values
fit_caret_LASSO
fit_caret_LASSO$results
fit_caret_LASSO$results$RMSE
min(fit_caret_LASSO$results$RMSE)
fit_caret_LASSO$bestTune
fit_caret_LASSO$finalModel

# cross-validation of kNN models (with standardization)
fit_caret_kNN = train(BodyFatSiri ~ . - Case - BodyFatBrozek - Density,
                      data = dataused,
                      method = "knn",
                      trControl = training,
                      preProcess = c("center","scale"),  # standardize the data
                      tuneGrid = expand.grid(k = allkNN))

# output 
fit_caret_kNN

############# identify selected model to fit to full data #############
# all best models
all_best_Types = c("Linear","Linear","LASSO","kNN")
all_best_Pars = list(3,14,fit_caret_LASSO$bestTune,fit_caret_kNN$bestTune)
all_best_Models = list(fit_caret_lm1$finalModel,
                       fit_caret_lm2$finalModel,
                       fit_caret_LASSO$finalModel,
                       fit_caret_kNN)
all_best_RMSE = c(fit_caret_lm1$results$RMSE,
                  fit_caret_lm2$results$RMSE,
                  min(fit_caret_LASSO$results$RMSE),
                  min(fit_caret_kNN$results$RMSE))

one_best_Type = all_best_Types[which.min(all_best_RMSE)]
one_best_Pars = all_best_Pars[which.min(all_best_RMSE)]
one_best_Model = all_best_Models[[which.min(all_best_RMSE)]]

############# one best model (LASSO) fit to full data #############
LASSOlambda = one_best_Pars[[1]]$lambda
LASSOcoef <- coef(one_best_Model,s=LASSOlambda)
#options(scipen=999)
#round(LASSOcoef,4)

############# compare all models - visual understanding #############
# model counts and types
mLin = length(allLinModels); mkNN = length(allkNN); mLASSO = length(alllambda)
mmodels = mLin+mkNN+mLASSO
modelMethod = c(rep("Linear",mLin),rep("LASSO",mLASSO),rep("kNN",mkNN))
all_caret_RMSE = c(fit_caret_lm1$results$RMSE,
                   fit_caret_lm2$results$RMSE,
                   fit_caret_LASSO$results$RMSE,
                   fit_caret_kNN$results$RMSE)
coloptions = rainbow(4)
colused = coloptions[as.numeric(factor(modelMethod))+1]
charused = 5*(as.numeric(factor(modelMethod)))
plot(1:mmodels,all_caret_RMSE,col=colused,pch=charused,
     xlab = "Model label",ylab = "RMSE")
order.min = c(1,2,2+which.min(fit_caret_LASSO$results$RMSE),
              2+mLASSO+which.min(fit_caret_kNN$results$RMSE))
abline(v=order.min,lwd=2)
abline(v=order.min[3],col="red",lwd=2)

