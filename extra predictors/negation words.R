library(tidyr)
library(tidytext)

# Problem: our joy, anger disgust sentiments do not take into word 
# negation into account. We will use 
# every pair of consecutive words to find the negated sentiments and score
# them according to the numeric Afinn sentiment lexicon. 

# This is a crude way
# to implement word negation bc we won't capture "not very good" versus "good"
# but we do filter out two-word negations.

#-------------------------------------------
# Define negation words
negations <- c("not", "no", "never", "without")

# Use tidytext fn unnest_tokens() to grab the `text` column and partition into
# 2-word tokens so "This place is trash" becomes "This place", "place is", 
# "is trash" 
bigrams <- yelp_train %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2)

# separate the 2-word bigrams into columns word1, word2
bigrams_sep <- bigrams %>%
  separate(bigram, c("word1", "word2"), sep = " ")

# How do we categorize the negated bigrams within our preexisting
# sentiment categories: "anger" "joy" etc.?

# I will use a different lexicon that weights sentiments numerically
# to simply flip the sign on negated sentiment words. However, the ratios
# are still capturing the good in "not good" rather than discounting them.

# Bring in afinn sentiments
afinn <- get_sentiments("afinn")

# ----------------------------------------

# 1. Use join the afinn sentiment scores to the bigrams_sep df
# (contains multiple rows per review bc each review is split into
# 2-word combos)

# 2. Mutate: set the negated bigrams to the opposite of its recognized
# sentiment, and the normal bigrams to the usual sentiment

# 3. Group_by and Summarise: there are too many observations per 
# review, so collapse the observations by review ID, then take their mean
# sentiment score.
afinn_scores <- bigrams_sep %>%
  inner_join(afinn, by = c(word2 = "word")) %>%
  mutate(ifelse(word1 %in% negations,
                -score, 
                score)) %>%
  group_by(X) %>%
  summarise(score = mean(score))


# JOIN THE AFINN SCORES TO YELP_TRAIN df
yelp_train <- left_join(yelp_train, afinn_scores, by = "X")




