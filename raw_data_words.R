# Learning from the raw data
library(tidytext)
library(dplyr)
library(magrittr)

# Get all the words from all the reviews in training data
allwords <- tolower(unlist(yelp_train$prunedtext))
allwords <- dropStopwords(allwords)

# Remove end of line characters and digits
allwords <- gsub("\t|\n|[[:digit:]]+", "", allwords)

length(unique(allwords))

df <- data.frame(table(allwords))

df %<>%
  arrange(desc(Freq))

# the most frequent "word" is empty space (created by the gsubs above)
# so skip over that one to get the top 2000 words
freqwords <- df[2:2002,]

# Get list of strings (freqwords is a 2-column df)
columns <- freqwords %>% 
  filter(allwords != "else") %>%
  pull(allwords)

# Create empty matrix with n rows and p columns (for 2000 most freq words)
x <- matrix(nrow = nrow(yelp_train), 
            ncol = 2000, 
            byrow = F)
colnames(x) <- columns # rename columns as the id'd words

# Run over the rows of our training data & save logical vector of
# which words in `column` are found in each row's list of words
for(r in 1:nrow(yelp_train)){
  bool <- columns %in% tolower(unlist(yelp_train$prunedtext[r]))
  x[r, ] <- bool # save the boolean vector into the rth row of x
}

raw_words_df <- as.data.frame(x)
colnames(raw_words_df) <- columns
raw_words_df$X <- rownames(raw_words_df) # add variable to join on

# Add number of stars categorical var to raw_words_num 
raw_words_num <- raw_words_df %>%
  mutate_if(is.logical, as.numeric)
raw_words_num$yelpstars <- as.factor(yelp_train$stars)

# Join by X
joined_words <- left_join(yelp_train, raw_words_num, by = "X")
joined_words %<>%
  mutate_if(is.logical, as.numeric)

write.csv(joined_words, "DATA/joined_words.csv")
write.csv(raw_words_num[1:20000,], "DATA/raw_words.csv")
#--------------------------------------------------
# NOT USED, but could be helpful if we want counts
# Input: row in the yelp_train data set (ie one review)
countWords <- function(string, row){
  bool <- columns %in% unlist(yelp_train$prunedtext[1])
  names(bool) <- columns
  ident_words <- columns[bool]
  for(c in ident_words){
    x[row, c] <- 1
    print(x[row,])
  }
}



set.seed(123)
rf <- randomForest(data = raw_words_num %>% select(-c(X)), 
                   yelpstars~., 
                   ntree = 100, 
                   mtry = ncol(raw_words_num%>% select(-c(X))) / 3)


