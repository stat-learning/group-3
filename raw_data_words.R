# Learning from the raw data
library(tidytext)
library(dplyr)
library(magrittr)

# Get all the words from all the reviews in training data
allwords <- tolower(unlist(yelp_train$prunedtext))
# Remove end of line characters and digits
allwords <- gsub("\t|\n|[[:digit:]]+", "", allwords)

allwords <- dropStopwords(allwords)
length(unique(allwords))


df <- data.frame(table(allwords))

# Arrange rows in df in descending order by word frequency
df %<>%
  arrange(desc(Freq))

# the most frequent "word" is empty space (created by the gsubs above)
# so skip over that one (start at row 2) to get the top 2000 words
freqwords <- df[1:2001,]

# Get list of strings (freqwords is a 2-column df)
columns <- freqwords %>% 
  filter(allwords != "else") %>%
  pull(allwords)
columns <- as.character(columns) # change from factor -> character

# Create empty counts matrix with n rows and p columns (for 2000 freq words)
mat.counts <- matrix(nrow = nrow(yelp_train), 
            ncol = 2000, 
            byrow = F)
colnames(mat.counts) <- columns # rename columns as the id'd words

# Run over the rows of our training data & save logical vector of
# which words in `column` are found in each row's list of words

#--------------------COUNT FUNCTIONS------------------------------
# Define helper function that actually counts # times each word appears
sumWords <- function(review, idwords){
  sum(review == idwords)
}
# Input: a row number in yelp_train
# Output: length 2000 vector, consisting of 0s and # time each of our top word appears
countWords <- function(row){
  review.words <- unlist(yelp_train$prunedtext[row]) # grabs list of words in review
  bool <- columns %in% review.words # columns - length 2000 (all 2000 most freq words)
  names(bool) <- columns
  ident.words <- columns[bool]
  sums <- sapply(ident.words, sumWords, 
                 review = review.words) # length is equal to 
  # the number of T cases in bool
  # slot the sums into the bool vector
  bool[bool==T] <- sums 
  bool 
}
#--------------------------------------------------

# Apply the above function over the matrix mat.counts (dims 2000 x 52,000)
for(r in 1:nrow(mat.counts)){
  mat.counts[r, ] <- countWords(r)
}
colSums(mat.counts)

# Save as data frame to then run PCA on
raw_words_df <- as.data.frame(mat.counts)
colnames(raw_words_df) <- columns
raw_words_df$X <- rownames(raw_words_df) # add variable to join on

# Add number of stars var to raw_words_num 
raw_words_df$response <- yelp_train$stars

# Join by X
joined_words <- left_join(yelp_train, raw_words_num, by = "X")
joined_words %<>%
  mutate_if(is.logical, as.numeric)

write.csv(joined_words, "DATA/joined_words.csv")
write.csv(raw_words_df, "DATA/raw_words.csv")







set.seed(123)
rf <- randomForest(data = raw_words_num %>% select(-c(X)), 
                   yelpstars~., 
                   ntree = 100, 
                   mtry = ncol(raw_words_num%>% select(-c(X))) / 3)


