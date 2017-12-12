require(knitr)

library(ISLR)
library(MASS)
library(caret)

attach(Boston)

Boston$resp <- "No"
Boston$resp[crim > median(crim)] <- 'Yes'
Boston$resp <-factor(Boston$resp)
table(Boston$resp)

Boston <- Boston[-drop(1)]

inTrain <- createDataPartition(y = Boston$resp, p = 0.75, list = FALSE)

train <- Boston[inTrain,]
test <- Boston[-inTrain,]

nzv <- nearZeroVar(train, saveMetrics = TRUE)
Cor <- cor(train[,-14])
highCor <- findCorrelation(Cor, cutoff = 0.75)

train_cor <- train[,-drop(c(2,9))]
test_cor <- test[,-drop(c(2,9))]

knnGrid <- expand.grid(.k=c(2))
# Use k = 2, since we expect 2 classes
KNN <- train(x=train_cor[,-12], method='knn',
             y=train_cor$resp, 
             preProcess=c('center', 'scale'), 
             tuneGrid = knnGrid)

