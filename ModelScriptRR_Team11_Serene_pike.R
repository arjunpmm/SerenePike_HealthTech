install.packages("Hmisc")
install.packages("ggpubr")
install.packages("corrplot")
install.packages("neuralnet")
install.packages("randomForest")

library(randomForest)
library(readxl)
library(ggpubr)
library(corrplot)
library(neuralnet)

# Visualisations 
plot(numerics$HR,type = "o", xlim=c(0,100), ylim = c(80,110))
#plot(numerics$RESP,type = "o", xlim=c(0, 375))

data <- numerics[c(3,4,5,6,8)]
#correlations
x <- as.data.frame(cor(data, method = c("pearson", "kendall", "spearman"), use = "complete.obs"))
corrplot(x, method = "circle")

#annova
summary(aov(RESP ~ numerics$Gender *numerics$Age, data=numerics))
ggboxplot(numerics, x = "Gender", y = "Age")


# Random sampling
samplesize = 0.70 * nrow(numerics)
set.seed(80)
index = sample( seq_len ( nrow ( numerics ) ), size = samplesize )

# Create training and test set
datatrain = numerics[ index, ]
datatest = numerics[ -index, ]

# fit neural network
set.seed(2)

# plot neural network
var1.matrix <- model.matrix(Gender~  - 1, data = datatrain)
datatrain2 <- data.frame(datatrain, var1.matrix)
nn <- neuralnet(RESP~ HR + PULSE + Age  + Gender +SpO2, datatrain, hidden=3, threshold=0.02, linear.output=FALSE)
plot(nn)
nn$result.matrix

predict_testNN = compute(nn, datatest[,c(3:7)])
predict_testNN = (predict_testNN$net.result * (max(datatest$RESP) - min(datatest$RESP))) + min(datatest$RESP)

plot(datatest$RESP, predict_testNN, col='blue', pch=16, ylab = "predicted NN", xlab = "real")

abline(0,1)

# Calculate Root Mean Square Error (RMSE)
RMSE.NN = (sum((datatest$RESP - predict_testNN)^2) / nrow(datatest)) ^ 0.5


# Function that returns Mean Absolute Error
mae <- function(error)
{
  mean(abs(error))
}


# Calculate error
error <- datatest$RESP - predict_testNN

# Example of invocation of functions
mae(error)

model1 <- randomForest(RESP ~ HR + PULSE + Age  + SpO2, data = datatrain, importance = TRUE)
model1
# Predicting on train set
predTrain <- predict(model1, datatrain, type = "class")
# Checking classification accuracy
table(predTrain, datatrain$RESP)  
predValid <- predict(model1, datatest, type = "class")
# Checking classification accuracy
mean(predValid == datatest$RESP)                    
table(predValid,datatest$RESP)
importance(model1)        
varImpPlot(model1) 



require(gbm)
require(MASS)#package with the boston housing dataset

Boston.boost=gbm(RESP ~ HR + PULSE + Age  + SpO2 + Gender,data = datatrain,distribution = "gaussian",n.trees = 10000,
                 shrinkage = 0.01, interaction.depth = 4)


summary(Boston.boost)
