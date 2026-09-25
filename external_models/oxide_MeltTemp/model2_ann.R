# install.packages("Metrics")
library(Metrics)
library(nnet)

data1 <- read.csv("## filepath ##\\model2_data.csv", encoding = "UTF-8")
# pred_ltcc <- read.csv("## filepath ##\\model2_ltcc_or_not.csv", encoding = "UTF-8")
# pred_ver <- read.csv("## filepath ##\\model2_verification.csv", encoding = "UTF-8")
pred_unknown <- read.csv("## filepath ##\\model2_unknown.csv", encoding = "UTF-8")


data <- data1[,-c(1:3)]
LLL <- dim(data)[2]
l <- dim(pred_unknown)[1]
data[,LLL+1] <- data1[,3]
names(data)[LLL+1] <- "ML"
data[,LLL+2] <- data1[,2]
names(data)[LLL+2] <- "Temperature"

names(pred_unknown)[3] <- "ML"

str(data)
str(pred_unknown)


feature <- read.csv("## filepath ##\\features+ML.csv", encoding = "UTF-8")
# feature <- read.csv("## filepath ##\\features.csv", encoding = "UTF-8")
names(feature)[1] <- "feature"
L <- dim(feature)[1]


# change the ## filepath ## into the real file path of the '.csv'
para <- read.csv("## filepath ##\\ann_para.csv", encoding = "UTF-8")
str(para)


pred <- matrix(0, nrow = l, ncol = 1000)

b = 31                                                                
a = 1

for (a in c(1:1000)){
  set.seed(a)
  par <- sample(2, nrow(data),replace = TRUE, prob = c(0.7,0.3))
  train <- data[par==1,]
  test <- data[par==2,]
  

  "formula" = as.character(feature[b,1])
  formula <- as.formula(formula)
  
  
  # optimized hyperparameters
  decay = para[b,2]
  size = para[b,3]
  
  set.seed(a)
  ann1=nnet(formula, data = train, maxit=3000,
            size=size, decay=decay, linout=T, trace=F)
  # print(rf)
  # summary(rf)
  # plot(rf)


  p <- predict(ann1, pred_unknown)
  #pp <- as.matrix(p)
  pred[,a] = p

  a = a+1
}

write.csv(pred,"C:\\Users\\Administrator.SC-201906181019\\Desktop\\S2_pred31.csv")




