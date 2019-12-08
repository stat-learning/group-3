library(MASS)
lda_model <- lda(starsfactor ~ ., data = num_only, na.action = "na.omit", CV = TRUE)
#LOOCV
conf <- table(num_only$starsfactor, lda_model$class)
misc <- 1 - (sum(diag(conf)) / sum(conf))
#0.5120608
#0.4753897 with negation

lda_crossval <- function(dat,k) {
  partition_index <- rep(1:k, each = nrow(dat)/k) %>% #Assigns a number to every obs in dat
    sample()
  dat <- dat[1:length(partition_index),] #prevents extraneous rows
  misclass_i <- vector(length = k) #preallocates memory for a vector
  dat$partition_index <- partition_index
  for (i in 1:k){
    test <- dat[which(dat$partition_index == i),] #get test data
    training <- dat[which(dat$partition_index != i),] #get training data
    lda_model <- lda(starsfactor ~ ., data = num_only, na.action = "na.omit")
    pred <- predict(object = lda_model, newdata = test, n.trees = 500, type = "response")
    conf <- table(test$starsfactor, pred$class)
    misclass_i[i] <- 1 - (sum(diag(conf)) / sum(conf))   #Fills in vector of length K
  }
  misclass <- mean(misclass_i)
  misclass
}

lda_crossval(num_only,5)
#0.511233

#qda_model <- lda(starsfactor ~ ., data = num_only, na.action = "na.omit")
qda_crossval <- function(dat,k) {
  partition_index <- rep(1:k, each = nrow(dat)/k) %>% #Assigns a number to every obs in dat
    sample()
  dat <- dat[1:length(partition_index),] #prevents extraneous rows
  misclass_i <- vector(length = k) #preallocates memory for a vector
  dat$partition_index <- partition_index
  for (i in 1:k){
    test <- dat[which(dat$partition_index == i),] #get test data
    training <- dat[which(dat$partition_index != i),] #get training data
    qda_model <- qda(starsfactor ~ ., data = num_only, na.action = "na.omit")
    pred <- predict(object = qda_model, newdata = test, n.trees = 500, type = "response")
    conf <- table(test$starsfactor, pred$class)
    misclass_i[i] <- 1 - (sum(diag(conf)) / sum(conf))   #Fills in vector of length K
  }
  misclass <- mean(misclass_i)
  misclass
}
qda_crossval(num_only,5)
#0.5630571
#0.547866 with negative words
ldap <- plot(lda_model)
plot(qda_model)
