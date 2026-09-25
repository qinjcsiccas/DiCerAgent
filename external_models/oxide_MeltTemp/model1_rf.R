library(Metrics)
library(randomForest)

data <- read.csv("## filepath ##\\model1_data.csv", encoding = "UTF-8")
# pred_ultcc <- read.csv("## filepath ##\\model1_ltcc_or_not.csv", encoding = "UTF-8")
# pred_ver <- read.csv("## filepath ##\\model1_verification.csv", encoding = "UTF-8")
pred_unknown <- read.csv("## filepath ##\\model1_unknown.csv", encoding = "UTF-8")



# head(data)
data[,1] <- as.character(data[,1])

row <- dim(pred_unknown)[1]
col <- dim(pred_unknown)[2]
L <- row

dataset <- data[,-c(1,2)]
d <- dim(dataset)[2]+1
dataset[,d] <- data[,2]
colnames(dataset)[d] <- "Temperature"


pred <- matrix(0, nrow = L, ncol = 1000)


a = 1
for (a in c(1:1000)){
  set.seed(a)
  
  ind<-sample(2,nrow(dataset),replace = T,prob = c(0.7,0.3))
  trainset<-dataset[ind == 1,]
  testset<-dataset[ind == 2,]

  rf <- randomForest(Temperature ~., trainset, ntree = 550, mtry = 3, maxnodes = 150)


  ptest <- predict(rf, pred_unknown)

  pp <- as.matrix(ptest)
  pred[,a] <- pp
  

  a = a+1
}


write.csv(pred,"## filepath ##\\S1_pred.csv")
