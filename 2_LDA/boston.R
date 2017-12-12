library(ISLR)
library(caret)
library(car)
library(MASS)
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
# data <- data.frame(scale(data)) #Scaling is done before learning by method train.
data$Class <- calculatedClass
data$Class <- as.factor(data$Class)
data <- data[,-1]
nrow(data[data$Class == 1,])


# Prepare training and test set.
TrainingDataIndex <- createDataPartition(data$Class, p=0.75, list=FALSE)
trainingData <- data[TrainingDataIndex,]
testData <- data[-TrainingDataIndex,]
trainingLabels <- trainingData$Class
testLabels <- testData$Class
testDataWithoutClass <- testData[,-14]
prop.table(table(trainingLabels))


## Find the correlations between variables.
nearZeroVariables <- nearZeroVar(trainingData, saveMetrics = TRUE)
correlations <- cor(trainingData[,-14])
highCorrelations <- findCorrelation(correlations, cutoff = 0.75)

trainingDataCorr <- trainingData[,-drop(c(4,9))]
testDataCorr <- testData[,-drop(c(4,9))]
testDataCorrWithoutClass <- testDataCorr[,-12]


# Logistic Regression
lr_model_all <- train(Class~., data=trainingData, 
                      method='glm', family=binomial(link='logit'),
                      preProcess=c('scale', 'center'))
lr_pred_all <- predict(lr_model_all, testData[,-14])
confusionMatrix(table(lr_pred_all, testData$Class))

lr_model_corr <- train(Class ~ rad + nox,
                       data=trainingData, 
                       method='glm', family=binomial(link='logit'),
                       preProcess=c('scale', 'center'))
lr_pred_corr <- predict(lr_model_corr, testData[,-14])
confusionMatrix(lr_pred_corr, testData$Class)
vif(lr_model_corr$finalModel)


# LDA
LDA_model_all <- train(Class~., data=trainingData,
                       method='lda', 
                       preProcess=c('scale', 'center'))
LDA_pred_all <- predict(LDA_model_all, testData[,-14])
confusionMatrix(LDA_pred_all, testData$Class)

LDA_model_corr <- train(Class~., data=trainingDataCorr,
             method='lda', 
             preProcess=c('scale', 'center'))
LDA_pred_corr <- predict(LDA_model_corr, testDataCorr[,-12])
confusionMatrix(LDA_pred_corr, testDataCorr$Class)


# k-NN
knnGrid <- expand.grid(.k=c(2))
knn_model_all <- train(x=trainingData[,-14], method='knn',
                       y=trainingData$Class, 
                       preProcess=c('center', 'scale'), 
                       tuneGrid = knnGrid)
knn_model_corr <- train(x=trainingDataCorr[,-12], method='knn',
             y=trainingDataCorr$Class, 
             preProcess=c('center', 'scale'), 
             tuneGrid = knnGrid)
knn_pred <- predict(knn_model_all, testData[,-14])
confusionMatrix(table(knn_pred, testLabels))
