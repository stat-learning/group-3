# BOOSTED MODEL: Seems like a good choice because the original random
# forest model is good at predicting the tail ends of the reviews, but 
# performs badly for reviews with 2-4 stars.

# Jacob's code:
#boosted model
#boost <- gbm(data = num_only, starsfactor ~ ., n.trees = 500, shrinkage = 0.01, interaction.depth = 1, cv.folds = 5, train.fraction = 0.2) #how to intepret results??
boost <- gbm(data = num_only, starsfactor ~ ., 
             n.trees = 100, 
             shrinkage = 0.1, 
             interaction.depth = 1) #how to intepret results

yhat.boost <- predict(boost, newdata = num_only, 
                      n.trees = 100, 
                      type = "response")

#boosted model
boost <- gbm(data = num_only, starsfactor ~ ., 
             n.trees = 500, 
             shrinkage = 0.01, 
             interaction.depth = 1) #how to intepret results

# Gives the important variables
summary(boost)

# Outputs a matrix of predicted values per class (k = 1,2,3,4,5 stars)
yhat.boost <- predict(boost, 
                      newdata = num_only, 
                      n.trees = 500, 
                      type = "response")

# Use function from lab 8 to place each of the 5 probabilities into one of the
# 5 categories (whichever predicted prob is largest) then calculate the 
# misclassification rate:
misClass <- function(model, data, n){
  yhats <- predict(model, newdata = data, 
                   type = "response", 
                   n.trees = n)
  predicted <- apply(yhats, 1, which.max)
  crosstab <- table(predicted, num_only$starsfactor) # gen misclass table
  correct <- diag(crosstab) 
  total <- table(num_only$starsfactor) 
  print(crosstab)
  (sum(total) - sum(correct)) / sum(total) # calc MSE
}

# Gives only the training error
misClass(boost, num_only, 500)



