# libraries used
library(FNN)
library(glmnet)

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
alllambda = exp(-100:20/10)
# kNN 
allkNN = 1:20
# model counts and types
mLin = length(allLinModels); mkNN = length(allkNN); mLASSO = length(alllambda)
mmodels = mLin+mkNN+mLASSO
modelMethod = c(rep("Linear",mLin),rep("LASSO",mLASSO),rep("kNN",mkNN))

############# Cross-validation via for loop #############
# produce loops for 10-fold cross-validation for model selection
nfolds = 10
groups = rep(1:nfolds,length=n)  #produces list of group labels
set.seed(81)
cvgroups = sample(groups,n)  #orders randomly

# set up storage for CV values
allmodelCV = rep(NA,mmodels) #place-holder for results

#fits for each model m, including predictors 1:m in order of forward step regression selection
# cross-validation of linear models
for (m in 1:mLin) {
  allpredictedCV = rep(NA,n)
  for (i in 1:nfolds) {
    groupi = (cvgroups == i)
    
    #prediction via cross-validation
    lmfitCV = lm(formula = allLinModels[[m]],data=bodyfat,subset=!groupi)
    allpredictedCV[groupi] = predict.lm(lmfitCV,bodyfat[groupi,])
    allmodelCV[m] = mean((allpredictedCV-bodyfat$BodyFatSiri)^2)
  }
}

# cross-validation of LASSO models
x = model.matrix(BodyFatSiri ~ . - Case - BodyFatBrozek - Density,data=bodyfat)[,-1]  # just select x variables
y = bodyfat$BodyFatSiri
cvLASSO = cv.glmnet(x, y, lambda=alllambda, alpha = 1, nfolds=nfolds, foldid=cvgroups)
allmodelCV[(mLin+1): (mLin+mLASSO)] = cvLASSO$cvm[order(cvLASSO$lambda)]

# cross-validation of kNN models (with standardization)
for (k in 1:mkNN) {
  allpredictedCV = rep(NA,n)
  for (i in 1:nfolds)  {
    groupi = (cvgroups == i)
    x.bodyfat = bodyfat[,5:16]  # just select x variables
    train.x = x.bodyfat[cvgroups != i,]
    train.x.std = scale(train.x)
    valid.x = x.bodyfat[cvgroups == i,]
    valid.x.std = scale(valid.x, 
                        center = attr(train.x.std, "scaled:center"), 
                        scale = attr(train.x.std, "scaled:scale"))
    predictedCV = knn.reg(train.x.std, valid.x.std, bodyfat$BodyFatSiri[!groupi], k = k)
    allpredictedCV[groupi] = predictedCV$pred
  }
  # must store in consecutive spots
  allmodelCV[mLin+mLASSO+k] = mean((allpredictedCV-bodyfat$BodyFatSiri)^2)
}

############# identify selected model to fit to full data #############
order.min = which.min(allmodelCV)
LinModel = ifelse(modelMethod[order.min] == "Linear", 
                 allLinModels[[order.min]],
                 "not best")
kNNpar = ifelse(modelMethod[order.min] == "kNN", 
                 allkNN[order.min-mLin], 
                 "not best")
LASSOpar = ifelse(modelMethod[order.min] == "LASSO",
  alllambda[order.min-mLin-mkNN], 
  "not best")

LinModel
kNNpar
LASSOpar

# then find coefficients to fit this best model
LASSOfit = glmnet(x, y, lambda=alllambda, alpha = 1)
LASSOcoef = coef(LASSOfit,s=LASSOpar)

############# compare all models #############
# compute RMSE = sqrt(CV) and plot
results = data.frame(RMSE = sqrt(allmodelCV),modelMethod=modelMethod)
coloptions = rainbow(4)
colused = coloptions[as.numeric(factor(modelMethod))+1]
charused = 5*(as.numeric(factor(modelMethod)))
plot(1:mmodels,results$RMSE,col=colused,pch=charused,
     xlab = "Model label",ylab = "RMSE")
abline(v=order.min,col="red")

