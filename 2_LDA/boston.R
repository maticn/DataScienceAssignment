library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization

library("MASS")
set.seed(7)
data <- Boston

# Calculate median crime rate and set class to each row regarding
# if it is above or below median crime rate. Scale the orher data.
median_crim_rate <- median(data$crim)
classFromCrimRate <- function(x) {
  if (x > median_crim_rate) return (1)
  else return (0)
}
calculatedClass <- sapply(data$crim, classFromCrimRate)
data <- data.frame(scale(data))
data$Class <- calculatedClass
data <- data[,-1]


# Prepare training and test set.
TrainingDataIndex <- createDataPartition(data$Class, p=0.75, list=FALSE)
trainingData <- data[TrainingDataIndex,]
testData <- data[-TrainingDataIndex,]
trainingLabels <- trainingData$Class
testLabels <- testData$Class
testDataWithoutClass <- testData[,-14]
prop.table(table(trainingLabels))


# Logistic Regression
lr_model <- glm(Class ~., family=binomial(logit), data=data)
summary(lr_model)
lr_pred <- predict(lr_model, testDataWithoutClass, type="response")

classFromProbabilities <- function(x) {
  if (x > 0.5) return (1)
  else return (0)
}
lr_pred_class <- sapply(lr_pred, classFromProbabilities)
confusionMatrix(table(lr_pred_class, testLabels))


# LDA
lda_model <- lda(Class ~., data=trainingData)
summary(lda_model)
lda_pred <- predict(lda_model, testDataWithoutClass)$class
confusionMatrix(table(lda_pred, testLabels))


# k-NN
decrementClass <- function(x) {
  if (x == 1) return (0)
  else if (x == 2) return (1)
  else return (-1)
}
k2_pred <- kmeans(testDataWithoutClass, centers = 2, nstart = 100)
k2_pred_corr <- sapply(k2_pred$cluster, decrementClass)
confusionMatrix(table(k2_pred_corr, testLabels))
