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
  trainx = model.matrix(BodyFatSiri ~ . - Case - BodyFatBrozek - Density, data = traindata)[,-1]
  trainy = traindata$BodyFatSiri
  validdata = bodyfat[groupj,]
  validx = model.matrix(BodyFatSiri ~ . - Case - BodyFatBrozek - Density, data = validdata)[,-1]
  validy = validdata$BodyFatSiri
  
  #specify data to be used
  dataused=traindata
  
  ###  entire model-fitting process ###
  ###  on traindata only!!! ###
  ###	 :	:	:	:	:	:	:   ###
  # set up training method
  set.seed(81)
  training = trainControl(method = "cv", number = 10)

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
       xlab = "Model label",ylab = "RMSE",
       ylim=c(min(all_caret_RMSE)+c(-.1,.5)))
  order.min = c(1,2,2+which.min(fit_caret_LASSO$results$RMSE),
                2+mLASSO+which.min(fit_caret_kNN$results$RMSE))
  abline(v=order.min,lwd=2)
  abline(v=which.min(all_caret_RMSE),col="red",lwd=2)
  
  one_best_Type = all_best_Types[which.min(all_best_RMSE)]
  one_best_Pars = all_best_Pars[which.min(all_best_RMSE)]
  one_best_Model = all_best_Models[[which.min(all_best_RMSE)]]

  ###  :	:	:	:	:	:	:   ###
  ###  resulting in     ###
  ###  one_best_Type and one_best_Pars and one_best_Model and one_best_Order  ###

  
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
                     "with parameter(s)",allbestPars[[j]])
  print(writemodel, quote = FALSE)
}

#assessment
y = bodyfat$BodyFatSiri
RMSE = sqrt(mean(allpredictedCV-y)^2); RMSE
R2 = 1-sum((allpredictedCV-y)^2)/sum((y-mean(y))^2); R2
# about 68.5% of the variability in BodyFatSiri values is 
# explained by this model-fitting process