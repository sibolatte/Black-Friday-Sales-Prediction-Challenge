#Black Friday Prediction Challenge

#A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month.
#The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month.
#Now, they want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products.

library(readr)
train <- read_csv("~/Desktop/Black Friday/train.csv")
View(train)

test <- read_csv("~/Desktop/Black Friday/test.csv")
View(test)

df_train<-train
df_test<-test
df_test$Purchase<-1 #adding a random numeric value to target variable so as to merge 2 datasets
comb<-rbind(df_train,df_test) #binding train and test

dim(comb) #display no. of rows and columns
summary(comb) #summary stats of the dataset


table(is.na(comb)) #checking the number of missing values in table
colSums(is.na(comb)) #columns that have these missing values # we see product_category2 and 3
#are the only variables with the missing values.

library(dplyr)
colextract<-select(comb,c(Product_ID,Product_Category_1,Product_Category_2,Product_Category_3))

#On selecting columns with missing values, it can be seen that category 2 and 3 have missing 
#values for the same products all across. For example, if there is a product ID P00085442 that
#is bought by multiple customers, its masked value in category 3 shows NA values.
# It means that product ID might not associated to category 3.So we can impute those missing values
#with 0 rather than removing the missing values.

#Imputing NAs with 0 in product_category_2 and product_category_3

comb$Product_Category_2[is.na(comb$Product_Category_2)]<-0
comb$Product_Category_3[is.na(comb$Product_Category_3)]<-0

#changing data type for variables
comb$Gender<-as.factor(comb$Gender)
comb$Age<-as.factor(comb$Age)
comb$Occupation<-as.factor(comb$Occupation)
comb$City_Category<-as.factor(comb$City_Category)
comb$Stay_In_Current_City_Years<-as.factor(comb$Stay_In_Current_City_Years)
comb$Marital_Status<-as.factor(comb$Marital_Status)
comb$Product_Category_1<-as.factor(comb$Product_Category_1)
comb$Product_Category_2<-as.factor(comb$Product_Category_2)
comb$Product_Category_3<-as.factor(comb$Product_Category_3)


#Graphic Visualization
#using only train dataset for exploring dataset visually.

library(ggplot2)
#plot(as.factor(df_train$Gender),df_train$Purchase,col="red")
ggplot(df_train,aes(y=df_train$Purchase,x=df_train$Gender))+geom_boxplot(fill="red")+coord_cartesian(ylim=c(0,30000))
#we see below males have purchased items more than females.
ggplot(df_train,aes(y=df_train$Purchase,x=df_train$Gender))+geom_bar(stat="identity",color="red")+ggtitle("Gender vs Purchase")
ggplot(df_train,aes(y=df_train$Purchase,x=df_train$Age))+geom_boxplot(fill="yellow")
ggplot(df_train,aes(y=df_train$Purchase,x=df_train$Age))+geom_bar(stat="identity",fill="yellow")
tapply(df_train$Purchase,df_train$Age,median)

ggplot(df_train,aes(y=df_train$Purchase,x=as.factor(df_train$Occupation)))+geom_boxplot(fill="blue")


ggplot(df_train,aes(y=df_train$Purchase,x=as.factor(df_train$City_Category)))+geom_boxplot(fill="orange")
ggplot(df_train,aes(y=df_train$Purchase,x=as.factor(df_train$Stay_In_Current_City_Years)))+geom_boxplot(fill="green")
ggplot(df_train,aes(y=df_train$Purchase,x=as.factor(df_train$Marital_Status)))+geom_boxplot(fill="pink")

######################Feature engineering#############################
f3<-aggregate(df_train$Purchase,list(df_train$Occupation,df_train$Age,df_train$Gender),median)
View(f3) #median purchase is calculated on basis of 3 features
colnames(f3)<-c("Occupation","Age","Gender","Median Purchase")
f3$GOA[f3$`Median Purchase`>8500]<-"Very High" # categorized median purchase as very high,medium and low
f3$GOA[f3$`Median Purchase`>7500 & f3$`Median Purchase`<=8500]<-"Medium"
f3$GOA[f3$`Median Purchase`<=7500]<-"Low"
f3$`Median Purchase`<-NULL

comb1<-comb
str(f3)
f3$Occupation<-as.factor(f3$Occupation)
f3$Gender<-as.factor(f3$Gender)
f3$Age<-as.factor(f3$Age)
f3$GOA<-as.factor(f3$GOA)
comb<-full_join(comb,f3,by=c("Occupation","Age","Gender")) #combined the "newly created categories" variable in main table 'comb'

f4<-aggregate(df_train$Purchase,list(df_train$Product_ID),median)
View(f4) #exploring data on basis of product, which product amounts to greater revenue

f5<-df_train%>%group_by(Product_ID)%>%tally()

f5<-cbind(f5,f4$Group.1,f4$x)
f5$`f4$Group.1`<-NULL

#categorized revenue as high,medium low generated from products
f5<-mutate(f5,RevenuefromProducts=f5$n*f5$`f4$x`)
quantile(f5$RevenuefromProducts,probs =c(0.1,0.3,0.5,0.7,0.8,0.9))
f5$TotRevenue[f5$RevenuefromProducts>20000000]<-"Very High"
f5$TotRevenue[f5$RevenuefromProducts>400000 & f5$RevenuefromProducts<20000000]<-"Medium"
f5$TotRevenue[f5$RevenuefromProducts<400000]<-"Low"

f5$n<-NULL
f5$`f4$x`<-NULL
f5$RevenuefromProducts<-NULL

comb<-full_join(comb,f5,by="Product_ID") #new column added to dataset


###################Modeling###############################################
comb1<-comb

TrainData<-comb[comb$Purchase!=1,]
TestData<-comb[comb$Purchase==1,]

View(TrainData)
View(TestData)

library(randomForest)
library(caret)
library(gbm)

TestData$TotRevenue[TestData$TotRevenue=='Other']="Low"

#fit gbm
TrainData$TotRevenue<-as.factor(TrainData$TotRevenue)
fitControl <- trainControl(method = "cv",number = 10)
tune_Grid <-  expand.grid(interaction.depth = 2,
                          n.trees = 100,
                          shrinkage = 0.1,
                          n.minobsinnode = 10)
fit <- train(Purchase ~ Gender + Age + Occupation + City_Category + Stay_In_Current_City_Years + 
               Marital_Status + Product_Category_1 + Product_Category_2 + Product_Category_3 + TotRevenue + GOA, data = TrainData,
             method = "gbm",
             trControl = fitControl,
             verbose = FALSE,
             tuneGrid = tune_Grid)

TestData$TotRevenue<-as.factor(TestData$TotRevenue)
Predicted= predict(fit,TestData)
Predicted=as.data.frame(Predicted)


Submission<-cbind(TestData$User_ID,TestData$Product_ID,Predicted)
colnames(Submission)[3]<-"Purchase"
write.csv(Submission,"/Users/sofiaarora/Desktop/sub1.csv",row.names=F)

#################applying second model##########################################

#fit h2o
install.packages("h2o")
library(h2o)
localH2O <- h2o.init(nthreads = -1)
h2o.init()
train.h2o <- as.h2o(TrainData)
test.h2o <- as.h2o(TestData)
colnames(train.h2o)
y.dep<-12
x.indep<-c(3:11,13:14)

system.time(gbm.model<-h2o.gbm(y=y.dep, x=x.indep, training_frame = train.h2o, ntrees = 2000, max_depth = 4, learn_rate = 0.01, seed = 1122))
predict.gbm <- as.data.frame(h2o.predict(gbm.model, test.h2o))

Submission<-cbind(TestData$User_ID,TestData$Product_ID,predict.gbm)
colnames(Submission)<-c("User_ID","Product_ID","Purchase")
write.csv(Submission,"/Users/sofiaarora/Desktop/sub4.csv",row.names=F)


#################applying third model##########################################

#fit deep learning
system.time(dlearning.model <- h2o.deeplearning(y = y.dep, x = x.indep,
                                      training_frame = train.h2o,
                                      epoch = 60,
                                      hidden = c(100,100),
                                      activation = "Rectifier",
                                      seed = 1122))
predict.dl2 <- as.data.frame(h2o.predict(dlearning.model, test.h2o))
Submission<-cbind(TestData$User_ID,TestData$Product_ID,predict.dl2)
colnames(Submission)<-c("User_ID","Product_ID","Purchase")
write.csv(Submission,"/Users/sofiaarora/Desktop/sub5.csv",row.names=F)

##################ensemble modeling############################################

#assign higher weight to the model with better accuracy
predict.ensemble<-0.3*predict.gbm + 0.7*predict.dl2
Submission<-cbind(TestData$User_ID,TestData$Product_ID,predict.ensemble)
colnames(Submission)<-c("User_ID","Product_ID","Purchase")
write.csv(Submission,"/Users/sofiaarora/Desktop/sub8.csv",row.names=F)

