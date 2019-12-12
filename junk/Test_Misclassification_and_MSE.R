#First run yelp_train through construction
#CLASSIFICATION
library(randomForest)
library(MASS)
library(gbm)

bool <- sapply(yelp_train, is.numeric)
num_only <- na.omit(yelp_train[,bool])
num_only$starsfactor <- as.factor(num_only$starsfactor)
set.seed(12)

#num_only <- select(num_only,-c("X", "X1", "stars")
num_only <- num_only[,-c(1:3,30)]

yelp_test <- read.csv("DATA/yelp_test_wpred.csv")
bool <- sapply(yelp_test, is.numeric)
test <- na.omit(yelp_test[,bool])
test$starsfactor <- as.factor(test$starsfactor)


#num_only <- select(num_only,-c("X", "X1", "stars")

#bagged trees:
bag <- randomForest(data = num_only, starsfactor~., ntree = 500, mtry = ncol(num_only) - 1)
pred_bag <- predict(object = bag, newdata = test, type = "response")
conf_bag <- table(test$starsfactor, pred_bag)
misc_bag <- 1 - (sum(diag(conf_bag)) / sum(conf_bag))
misc_bag

#type 1 error
t1bag <- c(1:5)
for (i in 1:5) {
  t1bag[i] <- conf_bag[i,i] / sum(conf_bag[,i])
}
t1_rate_bag <- mean(t1bag)

#type 2 error
t2bag <- c(1:5)
for (i in 1:5) {
  t2bag[i] <- conf_bag[i,i] / sum(conf_bag[i,])
}
t2_rate_bag <- mean(t2bag)

#random forest
rf <- randomForest(data = num_only, starsfactor~., ntree = 500, mtry = ncol(num_only)/3)
pred_rf <- predict(object = rf, newdata = test, type = "response")
conf_rf <- table(test$starsfactor, pred_rf)
misc_rf <- 1 - (sum(diag(conf_rf)) / sum(conf_rf))
misc_rf

#type 1 error
t1rf <- c(1:5)
for (i in 1:5) {
  t1rf[i] <- conf_rf[i,i] / sum(conf_rf[,i])
}
t1_rate_rf <- mean(t1rf)

#type 2 error
t2rf <- c(1:5)
for (i in 1:5) {
  t2rf[i] <- conf_rf[i,i] / sum(conf_rf[i,])
}
t2_rate_rf <- mean(t2rf)


#boosted trees
boost <- gbm(data = num_only, starsfactor ~ ., n.trees = 500, shrinkage = 0.1, interaction.depth = 1)
pred_boost <- predict(object = boost, newdata = test, type = "response", n.trees = 500)
predmat <- as.matrix(pred_boost[1:nrow(test),1:5,])
predclass <- colnames(predmat)[max.col(predmat,ties.method="first")]
conf_boost <- table(test$starsfactor, predclass)
misc_boost <- 1 - (sum(diag(conf_boost)) / sum(conf_boost))
misc_boost

#type 1 error
t1boost <- c(1:5)
for (i in 1:5) {
  t1boost[i] <- conf_boost[i,i] / sum(conf_boost[,i])
}
t1_rate_boost <- mean(t1boost)

#type 2 error
t2boost <- c(1:5)
for (i in 1:5) {
  t2bag[i] <- conf_boost[i,i] / sum(conf_boost[i,])
}
t2_rate_boost <- mean(t2boost)



#LDA
lda_model <- lda(starsfactor ~ ., data = num_only, na.action = "na.omit", CV = FALSE)

pred_lda <- predict(object = lda_model, newdata = test)
conf_lda <- table(test$starsfactor, pred_lda$class)
misc_lda <- 1 - (sum(diag(conf_lda)) / sum(conf_lda))
misc_lda

#type 1 error
t1lda <- c(1:5)
for (i in 1:5) {
  t1lda[i] <- conf_lda[i,i] / sum(conf_lda[,i])
}
t1_rate_lda <- mean(t1lda)

#type 2 error
t2lda <- c(1:5)
for (i in 1:5) {
  t2lda[i] <- conf_lda[i,i] / sum(conf_lda[i,])
}
t2_rate_lda <- mean(t2lda)


#QDA
qda_model <- qda(starsfactor ~ ., data = num_only, na.action = "na.omit", CV = FALSE)

pred_qda <- predict(object = qda_model, newdata = test)
conf_qda <- table(test$starsfactor, pred_qda$class)
misc_qda <- 1 - (sum(diag(conf_qda)) / sum(conf_qda))
misc_qda

#type 1 error
t1qda <- c(1:5)
for (i in 1:5) {
  t1qda[i] <- conf_qda[i,i] / sum(conf_qda[,i])
}
t1_rate_qda <- mean(t1qda)

#type 2 error
t2qda <- c(1:5)
for (i in 1:5) {
  t2qda[i] <- conf_qda[i,i] / sum(conf_qda[i,])
}
t2_rate_qda <- mean(t2qda)

#REGRESSION
bool <- sapply(yelp_train, is.numeric)
num_only <- na.omit(yelp_train[,bool])
num_only$stars <- as.numeric(num_only$starsfactor)
num_only <- num_only[,-c(1:2,8,30)]

set.seed(12)
bool <- sapply(yelp_test, is.numeric)
test <- na.omit(yelp_test[,bool])
test$stars <- as.numeric(test$starsfactor)

#bagged model
bag2 <- randomForest(data = num_only, stars~., ntree = 500, mtry = ncol(num_only) - 1)
pred_bag2 <- predict(object = bag2, newdata = test, type = "response")
adj_bag <- round(pred_bag2)
for (i in length(adj_bag)) {
  if (adj_bag[i] > 5) {
    adj_bag[i] <- 5
  }
  if (adj_bag[i] < 1) {
    adj_bag[i] <- 1
  }
}

mse_bag2 <- mean((adj_bag - test$stars)^2)
mse_bag2

#misclassification
conf_bag2 <- table(test$starsfactor, adj_bag)
misc_bag2 <- 1 - (sum(diag(conf_bag2)) / sum(conf_bag2))
misc_bag2

#type 1 error
t1bag2 <- c(1:5)
for (i in 1:5) {
  t1bag2[i] <- conf_bag2[i,i] / sum(conf_bag2[,i])
}
t1_rate_bag2 <- mean(t1bag2)

#type 2 error
t2bag2 <- c(1:5)
for (i in 1:5) {
  t2bag2[i] <- conf_bag2[i,i] / sum(conf_bag2[i,])
}
t2_rate_bag2 <- mean(t2bag2)

#random forest
rf2 <- randomForest(data = num_only, stars~., ntree = 500, mtry = ncol(num_only)/3)
pred_rf2 <- predict(object = rf2, newdata = test, type = "response")
adj_rf <- round(rf2)
for (i in length(adj_rf)) {
  if (adj_rf[i] > 5) {
    adj_rf[i] <- 5
  }
  if (adj_rf[i] < 1) {
    adj_rf[i] <- 1
  }
}

mse_bag2 <- mean((adj_rf- test$stars)^2)
mse_bag2



#misclassification
conf_rf2 <- table(test$starsfactor, adj_rf)
misc_rf2 <- 1 - (sum(diag(conf_rf2)) / sum(conf_rf2))
misc_rf2

#type 1 error
t1rf2 <- c(1:5)
for (i in 1:5) {
  t1rf2[i] <- conf_rf2[i,i] / sum(conf_rf2[,i])
}
t1_rate_rf2 <- mean(t1rf2)

#type 2 error
t2rf2 <- c(1:5)
for (i in 1:5) {
  t2rf2[i] <- conf_rf2[i,i] / sum(conf_rf2[i,])
}
t2_rate_rf2 <- mean(t2rf2)


#boosted trees
boost2 <- gbm(data = num_only, stars ~ ., n.trees = 500, shrinkage = 0.1, interaction.depth = 1)
pred_boost2 <- predict(object = boost2, newdata = test, type = "response", n.trees = 500)
adj_boost <- round(pred_boost2)
for (i in 1:length(adj_boost)) {
  if (adj_boost[i] > 5) {
    adj_boost[i] <- 5
  }
  if (adj_boost[i] < 1) {
    adj_boost[i] <- 1
  }
}

mse_boost2 <- mean((adj_boost - test$stars)^2)
mse_boost2

#misclassification
conf_boost2 <- table(test$stars, adj_boost)
misc_boost2 <- 1 - (sum(diag(conf_boost2)) / sum(conf_boost2))
misc_boost2

#type 1 error
t1boost2 <- c(1:5)
for (i in 1:5) {
  t1boost2[i] <- conf_boost2[i,i] / sum(conf_boost2[,i])
}
t1_rate_boost2 <- mean(t1boost2)

#type 2 error
t2boost <- c(1:5)
for (i in 1:5) {
  t2boost2[i] <- conf_boost2[i,i] / sum(conf_boost2[i,])
}
t2_rate_boost2 <- mean(t2boost2)



#linear regression
basiclinear <- lm(stars ~ joyratio + angerratio + fearratio + positiveratio + negativeratio + surpriseratio + disgustratio + trustratio + anticipationratio + nwords + n, data = num_only)
pred_linear <- predict(object = basiclinear, newdata = test, type = "response")
adj_linear <- round(pred_linear)
for (i in length(adj_linear)) {
  if (adj_linear[i] > 5) {
    adj_linear[i] <- 5
  }
  if (adj_linear[i] < 1) {
    adj_linear[i] <- 1
  }
}

mse_linear <- mean((adj_linear - test$stars)^2)
mse_linear

#misclassification
conf_linear <- table(test$starsfactor, adj_linear)
misc_linear <- 1 - (sum(diag(conf_linear)) / sum(conf_linear))
misc_linear

#type 1 error
t1linear <- c(1:5)
for (i in 1:5) {
  t1linear[i] <- conf_linear[i,i] / sum(conf_linear[,i])
}
t1_rate_linear <- mean(t1linear)

#type 2 error
t2linear <- c(1:5)
for (i in 1:5) {
  t2linear[i] <- conf_linear[i,i] / sum(conf_linear[i,])
}
t2_rate_linear <- mean(t2linear)


