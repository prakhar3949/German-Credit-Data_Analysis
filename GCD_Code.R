
library(dplyr)
library(glmnet)
library(ROCR)
library(PRROC)
library(boot)
library(rpart)
library(rpart.plot)

credit.data <- read.table("german.data")

colnames(credit.data)=c("chk_acct","duration","credit_his","purpose","amount","saving_acct","present_emp","installment_rate","sex","other_debtor","present_resid","property","age","other_install","housing","n_credits","job","n_people","telephone","foreign","response")

#orginal response coding 1= good, 2 = bad
#we need 0 = good, 1 = bad

credit.data$response = credit.data$response - 1

str(credit.data)

#converting response to factor

credit.data$response <- as.factor(credit.data$response)

summary(credit.data)

# Sampling data
set.seed(13254675)
index <- sample(nrow(credit.data),nrow(credit.data)*0.70)
credit.train = credit.data[index,]
credit.test = credit.data[-index,]

# Fitting full models with different links

credit.glm.logit<- glm(response~., family=binomial, data=credit.train)
credit.glm.probit<- glm(response~., data=credit.train,family=binomial(link = "probit"))
credit.glm.cloglog<- glm(response~., data=credit.train,family=binomial(link = "cloglog"))
summary(credit.glm.logit)

table.links<- data.frame("Name of the Link"=c("Logit","Probit","Cloglog"),
                  "Deviance"=c(credit.glm.logit$deviance,credit.glm.probit$deviance,credit.glm.cloglog$deviance),
                  "AIC"=c(AIC(credit.glm.logit),AIC(credit.glm.probit),AIC(credit.glm.cloglog)),
                  "BIC"=c(BIC(credit.glm.logit),BIC(credit.glm.probit),BIC(credit.glm.cloglog)))
table.links

# Probit has minimum Deviance, AIC and BIC values amongst the three Links. However, the differences 
# are very minute in values, hence we will stick with logit coz of easy of interpretibility

#AIC Variable Selection Technique(Backward selection)
credit.glm.back.AIC <- step(credit.glm.logit) # backward selection (if you don't specify anything)
summary(credit.glm.back.AIC)
credit.glm.back.AIC$deviance
AIC(credit.glm.back.AIC)
BIC(credit.glm.back.AIC)



#BIC Variable Selection Technique(Backward selection)
credit.glm.back.BIC <- step(credit.glm.logit, k=log(nrow(credit.train))) 
summary(credit.glm.back.BIC)
credit.glm.back.BIC$deviance
AIC(credit.glm.back.BIC)
BIC(credit.glm.back.BIC)


#LASSO Variable Selection

dummy<- model.matrix(~ ., data = credit.data)
credit.data.lasso<- data.frame(dummy[,-1])
credit.train.X = as.matrix(dplyr :: select(credit.data.lasso, -response1)[index,])
credit.test.X = as.matrix(dplyr ::select(credit.data.lasso, -response1)[-index,])
credit.train.Y = credit.data.lasso[index, "response1"]
credit.test.Y = credit.data.lasso[-index, "response1"]

#Lasso Model
credit.lasso<- glmnet(x=credit.train.X, y=credit.train.Y, family = "binomial")


##Perform cross-validation to determine the shrinkage parameter.
credit.lasso.cv<- cv.glmnet(x=credit.train.X, y=credit.train.Y, family = "binomial", type.measure = "class")

plot(credit.lasso.cv)


##For logistc regression, we can specify `type.measure="class"` so that the CV error will be misclassification error.

##Get the coefficient with optimal $\lambda$

coef(credit.lasso, s=credit.lasso.cv$lambda.min)
coef(credit.lasso, s=credit.lasso.cv$lambda.1se)

#in-sample predictions

# AIC model

pred.glm.a <- predict(credit.glm.back.AIC, type="response")


pred1 <- prediction(pred.glm.a, credit.train$response)
perf1 <- performance(pred1, "tpr", "fpr")
plot(perf1, colorize=TRUE)
AUC.a <- unlist(slot(performance(pred1, "auc"), "y.values"))
AUC.a

#misclassification rate table

#define a cost rate function
costfunc = function(obs, pred.p, pcut){
  weight1 = 5   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # count for "true=1 but pred=0"   (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost)
}
p.seq = seq(0.01, 1, 0.01) 

cost = rep(0, length(p.seq))  
for(i in 1:length(p.seq)){ 
  cost[i] = costfunc(obs = credit.train$response, pred.p = pred.glm.a, pcut = p.seq[i])  
}

plot(p.seq, cost)

optimal.pcut.glm.a = p.seq[which(cost==min(cost))]
optimal.pcut.glm.a

class.glm0.train.opt.a<- (pred.glm.a>optimal.pcut.glm.a)*1
table(credit.train$response, class.glm0.train.opt, dnn = c("True", "Predicted"))

MR.a<- mean(credit.train$response!= class.glm0.train.opt.a)
MR.a


# BIC Model
pred.glm.b <- predict(credit.glm.back.BIC, type="response")

pred2 <- prediction(pred.glm.b, credit.train$response)
perf2 <- performance(pred2, "tpr", "fpr")
plot(perf2, colorize=TRUE)
AUC.b <- unlist(slot(performance(pred2, "auc"), "y.values"))
AUC.b

#misclassification rate table


cost1 = rep(0, length(p.seq))  
for(i in 1:length(p.seq)){ 
  cost1[i] = costfunc(obs = credit.train$response, pred.p = pred.glm.b, pcut = p.seq[i])  
}

plot(p.seq, cost1)

optimal.pcut.glm.b = p.seq[which(cost1==min(cost1))]
optimal.pcut.glm.b

class.glm0.train.opt.b<- (pred.glm.b>optimal.pcut.glm.b)*1
table(credit.train$response, class.glm0.train.opt.b, dnn = c("True", "Predicted"))

MR.b<- mean(credit.train$response!= class.glm0.train.opt.b)
MR.b


# Lasso model
pred.lasso.train <- predict(credit.lasso, newx=credit.train.X, s=credit.lasso.cv$lambda.1se, type = "response")

pred3 <- prediction(pred.lasso.train, credit.train$response)
perf3<- performance(pred3, "tpr", "fpr")
plot(perf3, colorize=TRUE)
AUC.l <- unlist(slot(performance(pred3, "auc"), "y.values"))
AUC.l

#misclassification rate table

cost2 = rep(0, length(p.seq))  
for(i in 1:length(p.seq)){ 
  cost2[i] = costfunc(obs = credit.train$response, pred.p = pred.lasso.train, pcut = p.seq[i])  
}

plot(p.seq, cost2)

optimal.pcut.glm.l = p.seq[which(cost2==min(cost2))]
optimal.pcut.glm.l

class.glm0.train.opt.l<- (pred.lasso.train>optimal.pcut.glm.l)*1
table(credit.train$response, class.glm0.train.opt.l, dnn = c("True", "Predicted"))

MR.l<- mean(credit.train$response!= class.glm0.train.opt.l)
MR.l

# Comaprison Table

Comparison_table <- data.frame("Model" =c("AIC","BIC","Lasso"), "AUC" = c(AUC.a,AUC.b,AUC.l),
                    "MissClassification Rate"= c(MR.a,MR.b,MR.l),"Deviance"=c(credit.glm.back.AIC$deviance,credit.glm.back.BIC$deviance,NA),"No_of_Variables"=c(13,4,11))
Comparison_table

#Based on above table , we can say AIC model is best fit


#final model is
credit.glm.back.AIC 


#out-sample 

pred.glm.test<- predict(credit.glm.back.AIC, newdata = credit.test
                        [,-which(names(credit.test)=='response')], type="response")

#AUC
pred4 <- prediction(pred.glm.test, credit.test$response)
perf4 <- performance(pred4, "tpr", "fpr")
plot(perf4, colorize=TRUE)
AUC <- unlist(slot(performance(pred4, "auc"), "y.values"))
AUC

#MR
class.glm.test.opt<- (pred.glm.test>optimal.pcut.glm.a)*1
table(credit.test$response, class.glm.test.opt, dnn = c("True", "Predicted"))

MR<- mean(credit.test$response!= class.glm.test.opt)
MR

Comparison_table.test.train <- data.frame("Model" =c("Final_Model_Test","Final_Model_Train"), "AUC" = c(AUC,AUC.a),
                                      "MissClassification Rate"= c(MR,MR.a))
Comparison_table.test.train



#### Step 5 cross valiadation

pcut <-  1/6 
costfunc = function(obs, pred.p){
  weight1 = 5   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # count for "true=1 but pred=0"   (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
} # end of the function
 

credit.glm1 <- glm(response ~ chk_acct + duration + credit_his + purpose + 
                     amount + present_emp + installment_rate + other_debtor + 
                     age + other_install + n_credits + foreign, family = binomial, data=credit.data);  
cv.result = cv.glm(data=credit.data, glmfit=credit.glm1, cost=costfunc, K=3) 
cv.result$delta[2]


pred.glml <- predict(credit.glm1, newx=credit.data, type = "response")


pred5 <- prediction(pred.glml, credit.data$response)
perf5<- performance(pred5, "tpr", "fpr")
plot(perf5, colorize=TRUE)
AUC.cv <- unlist(slot(performance(pred5, "auc"), "y.values"))

#MR.cv
class.glm.cv<- (pred.glml>pcut)*1
table(credit.data$response, class.glm.cv, dnn = c("True", "Predicted"))

MR.cv<- mean(credit.data$response!= class.glm.cv)
MR.cv


Comparison_table.cv.aic <- data.frame("Model" =c("Final_Model_prediction_Test","CV"), "AUC" = c(AUC,AUC.cv),
                               "MissClassification Rate"= c(MR,MR.cv))
Comparison_table.cv.aic


## The above table states that AUC and MR are different for (iii) & (v). This proves that
## even though the model equation is same, by using the Cross-validation technique to divide
## the dataset into testing and training is a much better technique than randomly dividing the 
## dataset.


## CART

credit.rpart <- rpart(formula = response ~ . , data = credit.train, parms = list(loss=matrix(c(0,5,1,0), nrow = 2)))
summary(credit.rpart)

prp(credit.rpart,digits = 4, extra = 1)

#in sample
credit.train.pred.tree<- predict(credit.rpart, credit.train, type="class")
table(credit.train$response, credit.train.pred.tree, dnn=c("Truth","Predicted"))

# out of sample
credit.test.pred.tree<- predict(credit.rpart,
                                credit.test, type="class")
table(credit.test$response, credit.test.pred.tree, dnn=c("Truth","Predicted"))

MR.test.tree<- mean(credit.test$response!= credit.test.pred.tree)
MR.test.tree


#AUC
credit.test.prob.rpart = predict(credit.rpart,credit.test, type="prob")

pred.tree = prediction(credit.test.prob.rpart[,2], credit.test$response)
perf.tree = performance(pred.tree, "tpr", "fpr")
plot(perf.tree, colorize=TRUE)
AUC.tree <- slot(performance(pred.tree, "auc"), "y.values")[[1]]
AUC.tree

Comparison_table.final <- data.frame("Model" =c("Final_Model_prediction_Test","CV","Classification Tree Model"), "AUC" = c(AUC,AUC.cv,AUC.tree),
                                      "MissClassification Rate"= c(MR,MR.cv,MR.test.tree))
Comparison_table.final


# Logistic regression had MR 0.266 and AUC 0.844 where as 
# Tree classification has 0.396 and AUC 0.708 




