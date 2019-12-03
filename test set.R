# Test data set
library(dplyr)

path <- "/Users/ryankobler/Downloads/yelp_review.csv"

# Bring in old training data set
yelp_train <- read.csv("withlanguage.csv")

bigfile.sample <- read.csv(path,  
                           stringsAsFactors=FALSE, header=T, nrows=200000)  
cols <- colnames(bigfile.sample)
train <- yelp_train %>%
  select(cols)

# Use sample of data that does not include the training data set
no_train <- setdiff(bigfile.sample, train)
dim(no_train) - dim(bigfile.sample) 
# we've removed 1870 observations


# Take new sample of the data that has obs from training set removed
yelp_test <- no_train %>% 
  sample_frac(0.25)
dim(yelp_test)

write.csv(yelp_test, "DATA/yelp_test.csv")





