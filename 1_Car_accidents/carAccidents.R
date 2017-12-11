library(car)
library("caret")

setwd("D:/Dokumenti/R/1_data_science_lectures/Assignment/1_Car_accidents/")
filePath = "Data_Car_accidents_196.csv"
data <- read.csv(file=filePath, header=TRUE, sep=";", stringsAsFactors = TRUE)
data <- data[,2:6]
data$Gender <- as.factor(data$Gender)
data$Accident <- as.factor(data$Accident)
data$Socioeconomic_status <- as.factor(data$Socioeconomic_status)
summary(data)


### METHODS
getBetas <- function(logistic_model) {
  betas <- c()
  for (i in 1:length(logistic_model$coefficients)) { 
    betas <- c(betas, logistic_model$coefficients[i])
  }
  return (betas)
}

getOddsRatios <- function(betas) {
  oddsRatios <- c()
  for (i in 1:length(betas)) { 
    oddsRatios <- c(oddsRatios, exp(betas[i]))
  }
  return (oddsRatios)
}

getStandardErrors <- function(logistic_model) {
  standardErrors <- c()
  for (i in 1:length(summary(logistic_model)$coefficients[, 2])) { 
    standardErrors <- c(standardErrors, summary(logistic_model)$coefficients[, 2][i])
  }
  return (standardErrors)
}

getBounds <- function(oddsRatios, betas, standardErrors, confidence_number = 1.96) {
  bounds <- data.frame()
  for (i in 1:length(betas)) {
    low_bound <- exp(betas[i] - confidence_number * standardErrors[i])
    high_bound <- exp(betas[i] + confidence_number * standardErrors[i])
    entry <- data.frame(OR = oddsRatios[i], LB = low_bound, HB = high_bound, beta = betas[i], SE = standardErrors[i])
    bounds <- rbind(bounds, entry)
  }
  return (bounds)
}

plotBounds <- function(bounds, numf_of_columns_from_bounds = 3) {
  zeros <- rep(0, numf_of_columns_from_bounds)
  num_of_plots <- nrow(bounds)
  attach(mtcars)
  par(mfrow=c(num_of_plots, 1)) # number of rows and columns on the same graph
  
  for (i in 1:nrow(bounds)) {
    dots <- as.numeric(bounds[i,1:numf_of_columns_from_bounds])
    plot(dots, zeros, main=rownames(bounds)[i])
    #boxplot(dots)
  }
}
###


### 1) Identify demographic characteristics of the drivers that are risk
###    (or protective) factors of car accidents.
### a) crude (not adjusted) OR's
lm <- glm(Accident ~ BAC + Gender + Socioeconomic_status, family=binomial(logit), data=data)
summary(lm)
betas <- getBetas(lm)
oddsRatios <- getOddsRatios(betas)
standardErrors <- getStandardErrors(lm)
bounds <- getBounds(oddsRatios, betas, standardErrors)
plotBounds(bounds)


### b) adjusted OR's
lmAdj <- glm(Accident ~ BAC + Gender + Socioeconomic_status + Age, family=binomial(logit), data=data)
summary(lmAdj)
betasAdj <- getBetas(lmAdj)
oddsRatiosAdj <- getOddsRatios(betasAdj)
standardErrorsAdj <- getStandardErrors(lmAdj)
boundsAdj <- getBounds(oddsRatiosAdj, betasAdj, standardErrorsAdj)
plotBounds(boundsAdj)


### c) only BAC
lmBac <- glm(Accident ~ BAC, family=binomial(logit), data=data)
summary(lmBac)
betasBac <- getBetas(lmBac)
oddsRatiosBac <- getOddsRatios(betasBac)
standardErrorsBac <- getStandardErrors(lmBac)
boundsBac <- getBounds(oddsRatiosBac, betasBac, standardErrorsBac)
plotBounds(boundsBac)


### d) for fiveth...
lmFive <- glm(Accident ~ BAC + Gender + Age, family=binomial(logit), data=data)
summary(lmFive)
betasFive <- getBetas(lmFive)
oddsRatiosFive <- getOddsRatios(betasFive)
standardErrorsFive <- getStandardErrors(lmFive)
boundsFive <- getBounds(oddsRatiosFive, betasFive, standardErrorsFive)
plotBounds(boundsFive)


### 5) What is the probability that a 40 yr male whose BAC is >1%, causes a car accident?
###    What will be the probability, 10, 20, 30 and 40 years later? Is this change linear?
probabilityOfX <- function(BAC, Gender, Age, betas) {
  topFraction <- exp(betas[1] + betas[2] * BAC + betas[3] * Gender + betas[4] * Age)
  return (topFraction / (1 + topFraction))
}
p <- probabilityOfX(1, 1, 40, betasFive)
# p = 0.339
# BetaAge = 0.02245154 --> exp(BetaAge) = 1.022705
# Every year increase in age, makes response being true more likely with factor
# 1.022705, what is almost linear (but is not).

p10 <- probabilityOfX(1, 1, 50, betasFive)
p20 <- probabilityOfX(1, 1, 60, betasFive)
p30 <- probabilityOfX(1, 1, 70, betasFive)
p40 <- probabilityOfX(1, 1, 80, betasFive)


### 6) Evaluate the predictive performance of the model.
testFilePath = "Data_Car_accidents_17.csv"
testData <- read.csv(file=testFilePath, header=TRUE, sep=";", stringsAsFactors = TRUE)
testLabels <- testData[,3]
testData <- testData[,c("Gender", "Age", "Socioeconomic_status", "BAC")]
testData$Gender <- as.factor(testData$Gender)
testData$Socioeconomic_status <- as.factor(testData$Socioeconomic_status)
summary(testData)
predicted <- predict(lm, testData, type="response")

classFromProbabilities <- function(x) {
  if (x > 0.5) return (1)
  else return (0)
}

predictedClass <- sapply(predicted, classFromProbabilities)
performanceTable <- table(predictedClass, testLabels)
confussionMatrix <- confusionMatrix(performanceTable)

confussionMatrix
# precision = TP / (TP + FP)
