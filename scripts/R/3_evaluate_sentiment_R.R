# confusion matric using the r sentiment package, the package reads each day's headlines and gives a sentiment score to the daily news
# It is assumed that a positive sentiment score will be associated with a positive perspective in the market, and vice versa
# thus a confusion matrix will be used to show how accurate is the sentiment score in predicting the stock market





# change lagged adj change to binary response vector
dataset$laggedadjchange <- ifelse(dataset$laggedadjchange == 1, "increase", "decrease")



# get sentiment score for each day's news
sentimentscore_r <- rep(NA, dim(dataset)[1])
dataset <- cbind(dataset, sentimentscore_r)
for(i.day in 1:dim(dataset)[1]){
  dataset$sentimentscore_r[i.day] <- mean(sentiment(dataset$news[i.day])[[4]])
  if(i.day %% 100 == 0){
    cat(i.day, "/ 3653  >>  ")
  }
}



# Convert sentimentscore to factors with levels "increase" and "decrease"
predicted_labels <- as.factor(ifelse(dataset$sentimentscore_r > 0, "increase", "decrease"))



# Convert laggedadjchange to factors with levels "increase" and "decrease"
actual_labels <- as.factor(dataset$laggedadjchange)



# Create confusion matrix
cm <- confusionMatrix(predicted_labels, actual_labels,dnn = c("Prediction", "Reference"))



# plot confusion matrix
plt <- as.data.frame(cm$table)
plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
ggplot(plt, aes(Prediction,Reference, fill= Freq)) +
  geom_tile() + geom_text(aes(label=Freq)) + ggtitle("R sentiment confusion matrix") +
  scale_fill_gradient(low="white", high="#009194") +
  labs(x = "Reference",y = "Prediction") +
  scale_x_discrete(labels=c("Decrease","Increase")) +
  scale_y_discrete(labels=c("Increase","Decrease"))


