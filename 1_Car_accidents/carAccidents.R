library(car)
library("caret")
library("ggplot2")

# Read and prepare the data.
setwd("D:/Razvoj/DataScienceAssignment/1_Car_accidents")
filePath = "Data_Car_accidents_196.csv"
data <- read.csv(file=filePath, header=TRUE, sep=";", stringsAsFactors = TRUE)
data <- data[,2:6]  # remove IDs
data$Gender <- as.factor(data$Gender)
data$Accident <- as.factor(data$Accident)
data$Socioeconomic_status <- as.factor(data$Socioeconomic_status)
summary(data)

plot(data$Age, data$BAC, main="Age and BAC Scatter Plot", 
     xlab="Age", ylab="BAC", col=data$Accident, pch=19)


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

forestPlot <- function(df, boxLabels, yAxis, modelTitle) {
  #dev.off()
  p <- ggplot(df, aes(x = boxOdds, y = yAxis))
  p + geom_vline(aes(xintercept = 1), size = .25, linetype = "dashed") +
    geom_errorbarh(aes(xmax = boxCIHigh, xmin = boxCILow), size = .5, height = .2, color = "gray50") +
    geom_point(size = 3.5, color = "orange") +
    theme_bw() +
    theme(panel.grid.minor = element_blank()) +
    scale_y_continuous(breaks = yAxis, labels = boxLabels) +
    scale_x_continuous(breaks = seq(0,270,10) ) +
    #coord_trans(x = "log10") +
    ylab("") +
    xlab("Odds ratio (log scale)") +
    #annotate(geom = "text", y =1.1, x = 3.5, label ="Model p < 0.001\nPseudo R^2 = 0.10", size = 3.5, hjust = 0) +
    ggtitle(modelTitle)
}
###


### 1) Identify demographic characteristics of the drivers that are risk
###    (or protective) factors of car accidents.
onlyAccidents <- data[data$Acciden == 1, ]

gender <- data[data$Gender == 1, ]
genderAcc <- onlyAccidents[onlyAccidents$Gender == 1, ]
nrow(genderAcc)/nrow(gender)

genderF <- data[data$Gender == 0, ]
genderFAcc <- onlyAccidents[onlyAccidents$Gender == 0, ]
nrow(genderFAcc)/nrow(genderF)

ageLessThan41 <- data[data$Age < 41, ]
ageLessThan41Acc <- onlyAccidents[onlyAccidents$Age < 41, ]
nrow(ageLessThan41Acc)/nrow(ageLessThan41)

ageMoreThan40 <- data[data$Age > 40, ]
ageMoreThan40Acc <- onlyAccidents[onlyAccidents$Age > 40, ]
nrow(ageMoreThan40Acc)/nrow(ageMoreThan40)

status0 <- data[data$Socioeconomic_status == 0, ]
status0Acc <- status0[status0$Accident == 1, ]
nrow(status0Acc)/nrow(status0)

status1 <- data[data$Socioeconomic_status == 1, ]
status1Acc <- status1[status1$Accident == 1, ]
nrow(status1Acc)/nrow(status1)

status2 <- data[data$Socioeconomic_status == 2, ]
status2Acc <- status2[status2$Accident == 1, ]
nrow(status2Acc)/nrow(status2)


### 2) Obtain the model relating BAC and car accidents.
### a) Crude (not adjusted) OR's.
lmBac <- glm(Accident ~ BAC, family=binomial(logit), data=data)
summary(lmBac)
betasBac <- getBetas(lmBac)
oddsRatiosBac <- getOddsRatios(betasBac)
standardErrorsBac <- getStandardErrors(lmBac)
boundsBac <- getBounds(oddsRatiosBac, betasBac, standardErrorsBac)
plotBounds(boundsBac)

dev.off()
boxLabels = c("BAC")
yAxis = length(boxLabels):1
df <- data.frame(
  yAxis = length(boxLabels):1,
  boxOdds = c(boundsBac[2,1]),
  boxCILow = c(boundsBac[2,2]),
  boxCIHigh = c(boundsBac[2,3])
)
forestPlot(df, boxLabels, yAxis, "Crude (unadjusted) model")


### b) Adjusted OR's.
lmAdj <- glm(Accident ~ BAC + Gender + Socioeconomic_status + Age, family=binomial(logit), data=data)
summary(lmAdj)
betasAdj <- getBetas(lmAdj)
oddsRatiosAdj <- getOddsRatios(betasAdj)
standardErrorsAdj <- getStandardErrors(lmAdj)
boundsAdj <- getBounds(oddsRatiosAdj, betasAdj, standardErrorsAdj)
plotBounds(boundsAdj)

dev.off()
boxLabels = c("BAC", "Gender1", "Socioeconomic_status1", "Socioeconomic_status2", "Age")
yAxis = length(boxLabels):1
df <- data.frame(
  yAxis = length(boxLabels):1,
  boxOdds = c(boundsAdj[2,1],boundsAdj[3,1],boundsAdj[4,1],boundsAdj[5,1],boundsAdj[6,1]),
  boxCILow = c(boundsAdj[2,2],boundsAdj[3,2],boundsAdj[4,2],boundsAdj[5,2],boundsAdj[6,2]),
  boxCIHigh = c(boundsAdj[2,3],boundsAdj[3,3],boundsAdj[4,3],boundsAdj[5,3],boundsAdj[6,3])
)
forestPlot(df, boxLabels, yAxis, "Adjusted model")


### 5) What is the probability that a 40 yr male whose BAC is >1%, causes a car accident?
###    What will be the probability, 10, 20, 30 and 40 years later? Is this change linear?
probabilityOfX <- function(BAC, Gender, Age, betas) {
  topFraction <- exp(betas[1] + betas[2] * BAC + betas[3] * Gender + betas[6] * Age)
  return (topFraction / (1 + topFraction))
}
p <- probabilityOfX(1, 1, 40, betasAdj)
# p = 0.595
# BetaAge = 1.017061034 --> exp(BetaAge) = 2.765056
# Every year increase in age, makes response being true more likely with factor 2.765056.

p10 <- probabilityOfX(1, 1, 50, betasAdj)
p20 <- probabilityOfX(1, 1, 60, betasAdj)
p30 <- probabilityOfX(1, 1, 70, betasAdj)
p40 <- probabilityOfX(1, 1, 80, betasAdj)


### 6) Evaluate the predictive performance of the model.
testFilePath = "Data_Car_accidents_17.csv"
testData <- read.csv(file=testFilePath, header=TRUE, sep=";", stringsAsFactors = TRUE)
testLabels <- testData[,3]
testData <- testData[,c("Gender", "Age", "Socioeconomic_status", "BAC")]
testData$Gender <- as.factor(testData$Gender)
testData$Socioeconomic_status <- as.factor(testData$Socioeconomic_status)
summary(testData)
predicted <- predict(lmAdj, testData, type="response")

classFromProbabilities <- function(x) {
  if (x > 0.5) return (1)
  else return (0)
}

predictedClass <- sapply(predicted, classFromProbabilities)
performanceTable <- table(predictedClass, testLabels)
confussionMatrix <- confusionMatrix(performanceTable)

confussionMatrix
# precision = TP / (TP + FP)
