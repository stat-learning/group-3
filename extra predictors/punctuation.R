library(tokenizers)
library(dplyr)

yelp_train$prunedtext 
yelp_train$starsfactor <- as.factor(yelp_train$stars)

nSentences <- function(string){
  length(tokenize_sentences(string))
}

# Use tokenize_sentences() function to separate the long body of text into
# sentence segments. 
sentences <- tokenize_sentences(as.character(yelp_train$text))
yelp_train$sentences <- sapply(sentences, length)

# Stars appear to be decreasing slightly in sentence length
ggplot(data = yelp_train, aes(x = sentences, y = stars)) + 
  geom_jitter() + 
  geom_smooth()

# Distribution of sentence length
ggplot(data = yelp_train, aes(x = sentences)) +
  geom_bar()

#-----------------------------------------------------
# Punctuation mark frequency
library(stringr)

# Use the stringr library to count the number of punctuation marks of all
# types, designated by the "[[:punct:]]" pattern and save in new var `punct`.
yelp_train$punct <- str_count(as.character(yelp_train$text),
                              "[[:punct:]]")

yelp_train$exclaim <- str_count(as.character(yelp_train$text),
                              "!")

# Plot distribution of exclamation points
ggplot(data = yelp_train, aes(x = exclaim, y = stars)) + 
  geom_jitter() + 
  geom_smooth()

ggplot(data = yelp_train, aes(x = stars, y = punct)) + 
  geom_bar(stat = "identity") 


#rf <- randomForest(data = yelp_train, starsfactor~., 
                   #ntree = 500,
                   #mtry = ncol(num_only) / 3)











