
################## model design  ############################################################################

# response variable: sales, USD
# independent variables: adstocks of 7 channels in media part, activity_masks of 3 activities in trade part
# intercept: base sales
# 7 channels: TV,	Facebook,	Twitter,	Amazon,	Audio,	Print,	Digital_AO
# 3 activities: PriceChange,	Display,	EndCap
# number of coefficients that need to be estimated: 7+3+1=11
# root-mean-squre-error (RMSE) as a metric is used.
# linear regression will be used below at first to find the relationship between sales and 7 channels,3 activities, base sales.
# For the seasonal effect of base sales, bayesian hierarchical model (multilevel model) is used below via library(brms).

# summary from linear regression:
# (1) The contribution of base, trade and media on average are 80.63%, 7.86%, 11.50%, respectively.
# (2) 7 channels except Audio have positive effect on the total sales. The most and least effective channels are TV and
# Audio. Increasing 1 GRP in channels can add the total sales by $ 9343.95 where TV, Facebook, Twitter, Amazon,
# Audio, Print and Digital_AO contribute $ 2909.68, $ 2620.53, $ 2093.97, $ -1260.3, $ 598.85, $ 935.56 on average,respectively.
# (3) 3 activities except price change have positive effect on the total sales. Each 1% increase in the percentage of
# stores engaging in Display and EndCap may increase the total sales by $ 572.85 and $ 274.03 on average,
# respectively. Engaging in price change (=1) may decrease the total sales by $ 2791.75 on average.
# (4) More investment in channels except for Audio would be more likely to add the total sales. Without price change,
# trade activities (Display and EndCap) can increase the total sales.

# summary from bayesian hierarchical model (multilevel model):
# (1) mixed effect model via lmer from lme4 fails due to the singularity fit based on data given.
# (2) bayesian hierarchical model is fitted successfully, which can be checked by code below.
# (3) the seasonal effect of base sales can be represented by the random effect of the intercept in the model.
# (4) base sales (intercept) consists of the bias (fixed) part and seasonsal (random) part.
#     bias (fixed) part:   316690 (mean)
#     random part (spring):621.54
#     random part (summer):392
#     random part (fall):  -317.14
#     random part (winter):-373.41

# Part 2: compare models based on different machine learning methods to decide the best model. This part is optional, not required by competition.
# This part is shown in the appendix of solution code. Linear regression, ridge regression, Lasso, support vector machine (linear kernel),
# elastic net, Bayesian linear regression and random forest are used below to fit different learning models. The dataset
# is partitioned into training set (75%) and test set (25%). Root-mean-square-error (RMSE) is used as the performance
# metric based on test set.



### read datasets 
library(tidyverse)
library(readxl)

sales   = read_excel(file.choose(), sheet = "Sales")
activity= read_excel(file.choose(), sheet = "Trade Activities")
mask    = read_excel(file.choose(), sheet = "Trade Mask")
media   = read_excel(file.choose(), sheet = "Media")

n       = dim(sales)[1]                                                         # sample size = 148

### convert raw data of media table to data under GRP

media_c = media[,2:8]/ (1.28*10^6)                                              # operate on 2-8 columns of num type in media table
channel = dim(media_c)[2]                                                       # number of channels = 7
alpha   = 0.30                                                                  # assume decay parameter within (0,1)
# the best alpha (0.30~0.35) is searched via getting the min RMSE and max Rsquare value.
# more detailed can be seen in appendix below. 

### compute Adstock based on converted media table

library(phonTools)
adstock = as.data.frame(zeros(n,channel))                                       # adstock at each week for each channel
colnames(adstock) = colnames(media_c)                                           
adstock[1,] = media_c[1,]                                                       # in equation (1), A1 = Media_1, A0=0

for (i in 1:channel){                                                           # compute recursive At = Media_t + At-1 via loop
  adstock_t_1 = adstock[1,i]                                                   
  for (j in 2:n){
    adstock[j,i] = media_c[j,i] + alpha*adstock_t_1
    adstock_t_1  = adstock[j,i]
  }
}

### convert activity and mask data to one equivalent table : activity_mask

activity_mask = activity[,-1]*mask[,-1]                                         # do element-wise product for activity and mask tables


### linear regression
y = sales[,1]                                                                   # the response variable: sales
x = cbind(adstock,activity_mask)                                                # the independent variables: 7 vars from media, 3 from trade activity
dataset = cbind(y,x)
attach(dataset)

library(glmnet)                                                                 # generalized linear model package

linear_reg = lm(sales~TV+Facebook+Twitter+Amazon+Audio+Print+Digital_AO+PriceChange+Display+EndCap,data = dataset)
fit_y=predict(linear_reg,dataset)     
rmse = sqrt(mean((fit_y-dataset$sales)^2))                                      # training RMSE
summary(linear_reg)

#Coefficients:
#              Estimate   Std. Error t-value Pr(>|t|)    
#(Intercept)   316606.05   22948.46  13.796   <2e-16 ***
#  TV            2909.68      83.86  34.696   <2e-16 ***
#  Facebook      2620.53     173.48  15.106   <2e-16 ***
#  Twitter       2093.97    3490.23   0.600   0.5495    
#  Amazon         598.85     480.58   1.246   0.2149    
#  Audio        -1260.30    1409.10  -0.894   0.3727    
#  Print          935.56     423.60   2.209   0.0289 *  
#  Digital_AO    1445.66    1070.75   1.350   0.1792    
#  PriceChange  -2791.75    1756.38  -1.589   0.1143    
#  Display        572.85     606.68   0.944   0.3467    
#  EndCap         274.03     556.81   0.492   0.6234 


### adequacy of model
# 1.the fitted linear regression model is significant
# F-statistics = 310.8 shows the fitted model is significant.
# P-values of t-test shows some independent variables and intercept are significant.
# significant: intercept (base sales), TV, Facebook, Print, if p-value threshold alpha = 0.05.
# Rsquare = 0.9578 indicates that sales and independent variables may have linear relationships.

# 2. Residuals are Normally Distributed
plot(linear_reg)           # QQ-plot shows the residual has normal distribution
hist(linear_reg$residuals) # histogram shows the residual has normal distribution shape
library(fBasics)
jarqueberaTest(linear_reg$residuals) #Test residuals for normality via Jarque-Bera test
# Null Hypothesis: Skewness and Kurtosis are equal to zero.
# Residuals X-squared: 2.2435 p-Value: 0.3257 
# We fail to reject the Jarque-Bera null hypothesis (p-value = 0.3257).

# 3. Residuals are independent
library(lmtest) 
dwtest(linear_reg) #Durbin-Watson test
# DW = 1.9663, p-value = 0.2988.
# Null Hypothesis: Errors are serially UNcorrelated.
# We fail to reject the Durbin-Watson test's null hypothesis (p-value 0.2988).

# 4. Residuals have constant variance
# plot(linear_reg)
# the plot of residuals VS fitted values shows there may have about 2-3 outliers in this dataset.
# except for the 2-3 outilers, we can assume the residuals have constant variance.

#### contribution of base sales, Trade, media
#decay parameter = 0.30

# contribution of base sales
contr_base = linear_reg$coefficients[1]/linear_reg$fitted.values
min(contr_base)  # 58.85%
max(contr_base)  # 91.82%
mean(contr_base) # 80.63%

# contribution of trade
trade_cof = NULL
for (i in 1:148){
  trade_cof = rbind(trade_cof,linear_reg$coefficients[9:11])
}
contr_trade = dataset[,9:11]*trade_cof
contr_trade2 = rep(0,148)

for (i in 1:148){
  contr_trade2[i]=sum(contr_trade[i,])
}
contr_trade3 = contr_trade2/linear_reg$fitted.values
min(contr_trade3) # 5.86%
max(contr_trade3) # 9.76%
mean(contr_trade3) # 7.86%

# contribution of media
contr_media = (linear_reg$fitted.values - linear_reg$coefficients[1] - contr_trade2)/linear_reg$fitted.values
min(contr_media) # 0.21%
max(contr_media) # 34.88%
mean(contr_media) # 11.50%


### multilevel model or mixed effect model

# consider the base sales has the seasonal effect, 
# we can use mixed effect model and have random intercept effect, according to the seasons (spring, summer, fall, winter)

activity$season = rep(0,n)

for (i in 1:n){                                                                 # convert date into season factors
  week_ = as.data.frame(str_split(activity$Week[i], "-"))
  month = as.numeric(week_[2,])
  if (month >=1 & month <=3){
    activity$season[i] = 1 # spring
  }
  if (month >=4 & month <=6){
    activity$season[i] = 2 # summer
  }
  if (month >=7 & month <=9){
    activity$season[i] = 3 # fall
  }
  if (month >=10 & month <=12){
    activity$season[i] = 4 # winter
  }
  cat("i = ",i,"\t")
  print(activity$season[i])
}

dataset$season = activity$season
attach(dataset)

### mixed effect model
library(lme4)

mix_model = lmer(sales~TV+Facebook+Twitter+Amazon+Audio+Print+Digital_AO+PriceChange+Display+EndCap+(1 | season),data=dataset)
fit_y = predict(mix_model,dataset)     
rmse = sqrt(mean((fit_y-dataset$sales)^2))                                      # training RMSE = 9053.9
summary(mix_model) # singularity fit issue warning, this model fails

coef(mix_model)  #estimated coefficients in each group
fixef(mix_model) #average coefficients
ranef(mix_model) # group-level errors for the intercepts and slopes


### bayesian hierarchical model

library(brms)
#prior <- c(set_prior("normal(1000,10000)", class = "b"))
model1_brm=brm(sales~TV+Facebook+Twitter+Amazon+Audio+Print+Digital_AO+PriceChange+Display+EndCap+(1 | season), data=dataset
               ,iter=3000,chains=4,cores=4)
summary(model1_brm)
plot(model1_brm)

model1_brm$fit
# intercept has fixed part and random part from season:
# intercept fixed part: 316690.12 (mean),
# random part (spring):621.54
# random part (summer):392
# random part (fall):-317.14
# random part (winter):-373.41

conditional_effects(model1_brm)

################################ Appendix      ##########################################################################
### Tune decay parameter (alpha) in media execution model 
### to observe Rsquare value and MSE value in linear regression

Rsquare = NULL
RMSE = NULL
F_value = NULL

for (alpha in seq(0, 1, length = 21)) {                                         # decay parameter within (0,1), stepsize=0.05
### convert data of media table to Adstock data
library(phonTools)
adstock = as.data.frame(zeros(n,channel))                                       # adstock at each week for each channel
colnames(adstock) = colnames(media_c)                                           
adstock[1,] = media_c[1,]                                                       # in equation (1), A1 = Media_1, A0=0

for (i in 1:channel){                                                           # compute recursive At = Media_t + At-1 via loop
  adstock_t_1 = adstock[1,i]                                                   
  for (j in 2:n){
    adstock[j,i] = media_c[j,i] + alpha*adstock_t_1
    adstock_t_1  = adstock[j,i]
  }
}

### convert data of activity and mask to one equivalent table : activity_mask
activity_mask = activity[,-1]*mask[,-1]                                         # do element-wise product for activity and mask tables

### linear regression
y = sales[,1]
x = cbind(adstock,activity_mask)
dataset = cbind(y,x)
attach(dataset)

library(glmnet)
linear_reg = lm(sales~TV+Facebook+Twitter+Amazon+Audio+Print+Digital_AO+PriceChange+Display+EndCap,data = dataset)
metric = summary(linear_reg)
fit_y = predict(linear_reg,dataset)     
rmse = sqrt(mean((fit_y-dataset$sales)^2))  

Rsquare = cbind(Rsquare,metric$r.squared)
RMSE = cbind(MSE,rmse)
F_value = cbind(F_value,metric$fstatistic)

cat("alpha=",alpha,"\n")
cat("Rsquare=",metric$r.squared,"\n")
cat("RMSE=",rmse,"\n")
}

Rsquare
RMSE
plot(seq(0, 1, length = 21),Rsquare)
plot(seq(0, 1, length = 21),MSE)

# when decay parameter alpha = 0.30, Rsquare = 0.9577799 (secondary max), RMSE = 8844.404 (min)
# when decay parameter alpha = 0.35, Rsquare = 0.9597111 (max),           RMSE = 9247.767 (secondary min)


##################### compare performance of different methods  ############################################

# linear regression, ridge regression, Lasso, support vector machine (linear kernel), elastic net, 
# bayesian linear regression and random forest are used below to fit different learning models.
# the dataset is partitioned into training set (75%) and test set (25%).
# root-mean-square-error (RMSE) is used as the performance metric based on test set.

# from the results of RMSE below, bayesian linear regression (BLR) model has the best performance with min RMSE = 9415.619
# However, BLR only keep few independent variables. Similarly, Lasso and elastic net also remove 1-2 independent variables.


### partition dataset into training and test
library(caTools)
set.seed(101) 
sample = sample.split(dataset$sales, SplitRatio = 0.75)
train = subset(dataset, sample == TRUE)
test  = subset(dataset, sample == FALSE)

### linear regression performance
linear_reg = lm(sales~TV+Facebook+Twitter+Amazon+Audio+Print+Digital_AO+PriceChange+Display+EndCap,data = train)
predictedy = predict(linear_reg,test)
linear_reg_rmse = sqrt(mean((predictedy-test$sales)^2)) # test RMSE = 10260.51
summary(linear_reg)

### Ridge regression

set.seed(1011)
ridge_reg_cv = cv.glmnet(as.matrix(train[,-1]),as.matrix(train[,1]),alpha=0)
ridge_reg = glmnet(as.matrix(train[,-1]),as.matrix(train[,1]),alpha=0,lambda=ridge_reg_cv$lambda.min)

predictedy = predict(ridge_reg,s=ridge_reg_cv$lambda.min,newx=as.matrix(test[,-1]))
ridge_reg_rmse = sqrt(mean((predictedy-as.matrix(test[,1]))^2))   # test RMSE = 10519.3
ridge_reg$beta

### Lasso
set.seed(1011)
lasso_reg_cv = cv.glmnet(as.matrix(train[,-1]),as.matrix(train[,1]),alpha=1)
lasso_reg = glmnet(as.matrix(train[,-1]),as.matrix(train[,1]),alpha=1,lambda=lasso_reg_cv$lambda.min)

predictedy = predict(lasso_reg,s=lasso_reg_cv$lambda.min,newx=as.matrix(test[,-1]))
lasso_reg_rmse = sqrt(mean((predictedy-as.matrix(test[,1]))^2))   # test RMSE = 9897.074
lasso_reg$beta

### elastic net via ridge and lasso
# tune regularization parameter alpha which is different from the decay parameter in media data

elasticNet_RMSE = NULL
for (k in seq(0,1, length = 21)){
  set.seed(1011)
  elasticNet_reg_cv = cv.glmnet(as.matrix(train[,-1]),as.matrix(train[,1]),alpha=k)
  elasticNet_reg = glmnet(as.matrix(train[,-1]),as.matrix(train[,1]),alpha=k,lambda=elasticNet_reg_cv$lambda.min)
  
  predictedy = predict(elasticNet_reg,s=elasticNet_reg_cv$lambda.min,newx=as.matrix(test[,-1]))
  elasticNet_reg_rmse = sqrt(mean((predictedy-as.matrix(test[,1]))^2))   
  elasticNet_RMSE = cbind(elasticNet_RMSE,elasticNet_reg_rmse)
}


min(elasticNet_RMSE)  # min test RMSE = 9893.255, alpha = 0.90

set.seed(1011)
elasticNet_reg_cv = cv.glmnet(as.matrix(train[,-1]),as.matrix(train[,1]),alpha=0.90)
elasticNet_reg = glmnet(as.matrix(train[,-1]),as.matrix(train[,1]),alpha=0.90,lambda=elasticNet_reg_cv$lambda.min)

predictedy = predict(elasticNet_reg,s=elasticNet_reg_cv$lambda.min,newx=as.matrix(test[,-1]))
elasticNet_reg_rmse = sqrt(mean((predictedy-as.matrix(test[,1]))^2))   # test RMSE = 9893.255

elasticNet_reg$beta

### support vector machine
library(e1071)
library(Metrics)

svm_reg = svm(sales~TV+Facebook+Twitter+Amazon+Audio+Print+Digital_AO+PriceChange+Display+EndCap,data=train,kernel="linear",
             shrinking=TRUE,cross=5)
predictedy = predict(svm_reg,test)

svm_reg_rmse = rmse(test$sales,predictedy)        # test RMSE = 10219.51
summary(svm_reg)

### bayesian linear regression
library(BAS)

bayes_reg = bas.lm(sales~TV+Facebook+Twitter+Amazon+Audio+Print+Digital_AO+PriceChange+Display+EndCap,data=train,prior="AIC",pivot=TRUE)
predictedy = predict(bayes_reg,test,estimator="HPM")

bayes_reg_rmse = cv.summary.bas(predictedy$fit, test[,1])  # test RMSE = 10260.51

summary(bayes_reg)
plot(bayes_reg)
bayes_reg
image(bayes_reg)
bayes_reg_coef = coef(bayes_reg)
plot(bayes_reg_coef)
plot(confint(bayes_reg_coef,parm = 2:11))

# try diverse priors
bayes_RMSE = NULL
prior_list = c("BIC","AIC","g-prior","hyper-g","hyper-g-laplace","hyper-g-n",
                "JZS","ZS-null","ZS-full","EB-local","EB-global")

for (k in prior_list){
  
  bayes_reg = bas.lm(sales~TV+Facebook+Twitter+Amazon+Audio+Print+Digital_AO+PriceChange+Display+EndCap,data=train,prior=k,pivot=TRUE)
  predictedy = predict(bayes_reg,test,estimator="HPM")
  
  bayes_reg_rmse = cv.summary.bas(predictedy$fit, test[,1])
  bayes_RMSE = cbind(bayes_RMSE,bayes_reg_rmse)
  
}

min(bayes_RMSE)  # min test RMSE = 9415.619, prior = "g-prior", but this model removes many independent variables


### random forest

library(randomForest)
set.seed(1)

randomF_reg = randomForest(sales~TV+Facebook+Twitter+Amazon+Audio+Print+Digital_AO+PriceChange+Display+EndCap,data=train,mtry=10,improtance=TRUE)
predictedy = predict(randomF_reg,newdata = test)

randomF_reg_rmse = sqrt(mean((predictedy -test[,1])^2))    # test RMSE = 15015.04

importance(randomF_reg)
varImpPlot(randomF_reg)
summary(randomF_reg)

