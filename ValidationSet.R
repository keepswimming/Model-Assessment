############# libraries ############
library(dplyr)
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

################################################################
##### Validation set assessment of entire modeling process #####				 
################################################################

##### model assessment outer validation shell #####
#define the split into training set (typically of size about 2/3 of data) and validation set (of size about 1/3)
n.train = round(n*2/3); n.train
n.valid = n-n.train; n.valid
set.seed(82)
whichtrain = sample(1:n,n.train)  #produces list of data to use in training
include.train = is.element(1:n,whichtrain)  # sets up a T-F vector to be used similarly as group T-F vectors
include.valid = !is.element(1:n,whichtrain)  # sets up a T-F vector to be used similarly as group T-F vectors

#just one split into training and validation sets
traindata = bodyfat[include.train,]
trainx = model.matrix(BodyFatSiri ~ . - Case - BodyFatBrozek - Density, data = traindata)[,-1]
trainy = traindata$BodyFatSiri
validdata = bodyfat[include.valid,]
validx = model.matrix(BodyFatSiri ~ . - Case - BodyFatBrozek - Density, data = validdata)[,-1]
validy = validdata$BodyFatSiri

#specify data to be used
dataused=traindata

###  entire model-fitting process ###
###  on traindata only!!! ###
###	 :	:	:	:	:	:	:   ###

# set up training method
set.seed(82)
training = trainControl(method = "cv", number = 10)

# cross-validation of linear model 1
fit_caret_lm1 = train(BodyFatSiri ~ Weight+BMI+Height,
                      data = dataused,
                      method = "lm",
                      trControl = training)

# cross-validation of linear model 2
fit_caret_lm2 = train(BodyFatSiri ~ . - Case - BodyFatBrozek - Density,
                      data = dataused,
                      method = "lm",
                      trControl = training)

# cross-validation of LASSO
fit_caret_LASSO = train(BodyFatSiri ~ . - Case - BodyFatBrozek - Density,
                        data = dataused,
                        method = "glmnet",
                        trControl = training,
                        tuneGrid = expand.grid(alpha=c(1),lambda=alllambda))

# cross-validation of kNN models (with standardization)
fit_caret_kNN = train(BodyFatSiri ~ . - Case - BodyFatBrozek - Density,
                      data = dataused,
                      method = "knn",
                      trControl = training,
                      preProcess = c("center","scale"),
                      tuneGrid = expand.grid(k = allkNN))

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

###  :	:	:	:	:	:	:   ###
###  resulting in     ###
###  one_best_Type and one_best_Pars and one_best_Model and one_best_Order  ###

if (one_best_Type == "Linear") {  # then best is one of linear models
  allpredictedvalid = predict(one_best_Model,validdata)
} else if (one_best_Type == "LASSO") {  # then best is one of LASSO models
  LASSOlambda = one_best_Pars[[1]]$lambda
  allpredictedvalid  = predict(one_best_Model,newx=validx,s=LASSOlambda)
} else if (one_best_Type == "kNN") {  # then best is one of kNN models
  allpredictedvalid = one_best_Model %>% predict(validdata)
}

plot(allpredictedvalid,validy)
RMSE = sqrt(mean(allpredictedvalid-validy)^2); RMSE
R2 = 1-sum((allpredictedvalid-validy)^2)/sum((validy-mean(validy))^2); R2
# About 70% of variability in truly new predictions is explained by this modeling process.