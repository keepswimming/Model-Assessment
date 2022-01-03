############# library ############
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
alllambda = exp(-100:15/10)
# kNN 
allkNN = 1:20

###################################################################
##### Double cross-validation for modeling-process assessment #####				 
###################################################################

##### model assessment OUTER shell #####
# produce loops for 10-fold cross-validation for model ASSESSMENT
nfolds = 10
groups = rep(1:nfolds,length=n)  #produces list of group labels
set.seed(82)
cvgroups = sample(groups,n)  #orders randomly

# set up storage for predicted values from the double-cross-validation
allpredictedCV = rep(NA,n)
# set up storage to see what models are "best" on the inner loops
allbestTypes = rep(NA,nfolds)
allbestPars = vector("list",nfolds)

# loop through outer splits
for (j in 1:nfolds)  {  #be careful not to re-use loop indices
  groupj = (cvgroups == j)
  traindata = bodyfat[!groupj,]
  trainx = model.matrix(BodyFatSiri ~ . - Case - BodyFatBrozek - Density, data = traindata)[,-(1:4)]
  trainy = traindata$BodyFatSiri
  validdata = bodyfat[groupj,]
  validx = model.matrix(BodyFatSiri ~ . - Case - BodyFatBrozek - Density, data = validdata)[,-(1:4)]
  validy = validdata$BodyFatSiri
  
  #specify data to be used
  dataused=traindata

  ###  entire model-fitting process ###
  ###  on traindata only!!! ###
  ###	 :	:	:	:	:	:	:   ###
  ###INCLUDING ALL CONSIDERED MODELS###				 
  ###  :	:	:	:	:	:	:   ###
  ###  resulting in     ###
  ###  one_best_Type and one_best_Pars and one_best_Model ###
  
  allbestTypes[j] = one_best_Type
  allbestPars[[j]] = one_best_Pars
  
  if (one_best_Type == "Linear") {  # then best is one of linear models
    allpredictedCV[groupj] = predict(one_best_Model,validdata)
  } else if (one_best_Type == "LASSO") {  # then best is one of LASSO models
    LASSOlambda = one_best_Pars[[1]]$lambda
    allpredictedCV[groupj]  = predict(one_best_Model,newx=validx,s=LASSOlambda)
  } else if (one_best_Type == "kNN") {  # then best is one of kNN models
    allpredictedCV[groupj] = one_best_Model %>% predict(validdata)
  }
}

# for curiosity / consistency, we can see the models that were "best" on each of the inner splits
allbestTypes
allbestPars
# print individually
for (j in 1:nfolds) {
  writemodel = paste("The best model at loop", j, 
                     "is of type", allbestTypes[j],
                     "with parameter(s)",allbestPars[j])
  print(writemodel, quote = FALSE)
}

#assessment
y = bodyfat$BodyFatSiri
RMSE = sqrt(mean(allpredictedCV-y)^2); RMSE
R2 = 1-sum((allpredictedCV-y)^2)/sum((y-mean(y))^2); R2
