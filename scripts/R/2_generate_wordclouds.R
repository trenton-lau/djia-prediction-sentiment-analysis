# 2.2 serves as an supplementary part to the NLP, it reveals how sentiment affects the stock price and to what extent it influence
# It is first modified using tm to corpus and punctuation is removed, bing lexicon dictionary is used which consists of 6000+ English words with either positive or negative sentiment.
# It reveals the top ten positive or negative words that appear in the news headlines





# modify text
corpus <- VCorpus(VectorSource(data[,4]))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords())
corpus <- tm_map(corpus, stemDocument)
corpus <- tm_map(corpus, stripWhitespace)



# create matrix of word count
dtm <- DocumentTermMatrix(corpus)
word_count <- as.data.frame(as.matrix(dtm))
word_count <- apply(word_count, 2, sum)
word_count <- cbind(names(word_count), word_count)
colnames(word_count) <- c("word", "count")
rownames(word_count) <- 1:dim(word_count)[1]
word_count[,1] <- str_replace_all(word_count[,1], "[[:punct:]]", " ")
word_count <- as.data.frame(word_count)



# bing sentiment analysis
sentiment_count <- word_count %>%
  inner_join(get_sentiments("bing"), by = "word")
sentiment_count



# subset for positive and negative sentiment
neg_sentiment <- subset(sentiment_count, subset = sentiment_count$sentiment == "negative")
pos_sentiment <- subset(sentiment_count, subset = sentiment_count$sentiment == "positive")
neg_sentiment$count <- as.numeric(as.character(neg_sentiment$count))
pos_sentiment$count <- as.numeric(as.character(pos_sentiment$count))



# select the top 10 for each sentiment
pos_sentiment_10 <- pos_sentiment[rev(order(pos_sentiment$count)),] %>% head(10)
neg_sentiment_10 <- neg_sentiment[rev(order(neg_sentiment$count)),] %>% head(10)
pos_sentiment_10 <- as.data.frame(pos_sentiment_10[,1:2])
neg_sentiment_10 <- as.data.frame(neg_sentiment_10[,1:2])



# plot
pos_sentiment_10 %>%
  mutate(word = fct_reorder(word, count)) %>%
  ggplot(aes(x = word, y = count)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  ggtitle("Positive")
neg_sentiment_10 %>%
  mutate(word = fct_reorder(word, count)) %>%
  ggplot(aes(x = word, y = count)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  ggtitle("Negative")



# word cloud
set.seed(3011)
neg_sentiment <- neg_sentiment[rev(order(neg_sentiment$count)),]
pos_sentiment <- pos_sentiment[rev(order(pos_sentiment$count)),]
neg_cloud <- wordcloud(words = neg_sentiment$word, freq = neg_sentiment$count, min.freq = 1,
                       max.words=500, random.order=FALSE, rot.per=0.35,
                       colors=brewer.pal(8, "Dark2"))
set.seed(3011)
pos_cloud <- wordcloud(words = pos_sentiment$word, freq = pos_sentiment$count, min.freq = 1,
                       max.words=500, random.order=FALSE, rot.per=0.35,
                       colors=brewer.pal(8, "Dark2"))


