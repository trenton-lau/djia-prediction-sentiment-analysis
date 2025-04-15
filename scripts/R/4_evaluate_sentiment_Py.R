# The confusion matrix is also used for the Python sentiment score done in the book, it is imported as sentiment.csv
# Notices that this sentiment.csv is given by Trenton LAU and Zheming CAO as they replicate the sentiment of the Python code for their LSTM part
# The confusion matrix is constricted to compare the R sentiment package and Python sentiment package





# import py sentiment data
sentiment <- read_excel("sentiment.csv")
sentimentscore_py <- sentiment$compound
dataset <- cbind(dataset, sentimentscore_py)



# change lagged adj change to binary response vector
dataset$laggedadjchange <- ifelse(dataset$laggedadjchange == 1, "increase", "decrease")



# Convert sentimentscore to factors with levels "increase" and "decrease"
predicted_labels <- as.factor(ifelse(dataset$sentimentscore_py > 0, "increase", "decrease"))



# Convert laggedadjchange to factors with levels "increase" and "decrease"
actual_labels <- as.factor(dataset$laggedadjchange)



# Create confusion matrix
cm <- confusionMatrix(predicted_labels, actual_labels, dnn = c("Prediction", "Reference"))



# plot confusion matrix
plt <- as.data.frame(cm$table)
plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
ggplot(plt, aes(Prediction,Reference, fill= Freq)) +
  geom_tile() + geom_text(aes(label=Freq)) + ggtitle("Python sentiment confusion matrix") +
  scale_fill_gradient(low="white", high="#009194") +
  labs(x = "Reference",y = "Prediction") +
  scale_x_discrete(labels=c("Decrease","Increase")) +
  scale_y_discrete(labels=c("Increase","Decrease"))


