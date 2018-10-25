#Loan Prediction

#importing packages
if(!require(data.table) | !require(mice) | !require(caTools) | !require(kernlab) | !require(class) | !require(e1071)
   | !require(rpart) | !require(randomForest) | !require(xgboost) | !require(gbm) | !require(caret) | !require(DMwR) 
   | !require(missForest) | !require(Amelia)){
  install.packages(c('rpart','randomForest', 'xgboost', 'gbm', 'caret','data.table','mice','caTools',
                     'Kernlab','class','e1071','DMwR','Amelia','missForest'))
}


#importing data
library(data.table)
train_data <- fread(file.choose(), stringsAsFactors = T, na.strings = c(""," ",NA))

head(train_data)
sapply(train_data, class)

#checking Missing values
sort(sapply(train_data, function(x) { sum(is.na(x)) }), decreasing=TRUE) #Something wrong at this point

#visualising Missing data
install.packages("VIM")
library(VIM)
mice_plot <- aggr(train_data, col=c('navyblue','yellow'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(train_data), cex.axis=.7,
                  gap=3, ylab=c("Missing data","Pattern"))   

#Imputing Missing Values for training set
library(mice)
imputed_data <- mice(train_data[,c(9,10,11)], m=5, maxit = 50, method = 'rf', seed = 500)
train_data[,c(9,10,11)] <- complete(imputed_data, 2)

#Encoding Categorical variables into Numeric for training data
train_data$Gender <- factor(train_data$Gender, labels = c(1,2), levels = c('Female','Male'))
train_data$Married <- factor(train_data$Married, labels = c(0,1), levels = c('No','Yes'))
train_data$Education <- factor(train_data$Education, labels = c(1,2), levels = c('Graduate','Not Graduate') )
train_data$Self_Employed <- factor(train_data$Self_Employed, labels = c(1,2), c('Yes','No'))
train_data$Property_Area <- factor(train_data$Property_Area, labels = c(1,2,3), levels = c('Rural','Urban','Semiurban'))
train_data$Loan_Status <- factor(train_data$Loan_Status, labels = c(0,1), levels = c('N','Y') )

#set column level
levels(train_data$Dependents)[levels(train_data$Dependents) ==  "3+"] <- "3"

#Imputing Missing Value for Gender and Married
imputed_data <- mice(train_data[,c(2,3,6)], m=5, maxit = 50, method = 'rf', seed = 500)
train_data[,c(2,3,6)] <- complete(imputed_data, 2)

#Converting categoical to integer for train data
train_data$Gender <- as.integer(train_data$Gender)
train_data$Loan_ID <- as.integer(train_data$Loan_ID)
train_data$Married <- as.integer(train_data$Married)
train_data$Dependents <- as.integer(train_data$Dependents)
train_data$Education <- as.integer(train_data$Education)
train_data$Self_Employed <- as.integer(train_data$Self_Employed)
train_data$Property_Area <- as.integer(train_data$Property_Area)

#feature Scaling
train_data$Loan_ID <- scale(train_data$Loan_ID)
train_data$Gender <- scale(train_data$Gender)
train_data$Married <- scale(train_data$Married)
train_data$Dependents <- scale(train_data$Dependents)
train_data$Education <- scale(train_data$Education)
train_data$Self_Employed <- scale(train_data$Self_Employed)
train_data$ApplicantIncome <- scale(train_data$ApplicantIncome)
train_data$CoapplicantIncome <- scale(train_data$CoapplicantIncome)
train_data$LoanAmount <- scale(train_data$LoanAmount)
train_data$Loan_Amount_Term <- scale(train_data$Loan_Amount_Term)
train_data$Credit_History <- scale(train_data$Credit_History)
train_data$Property_Area <- scale(train_data$Property_Area)

#Outlier Detection
library(DMwR)
outlier.scores <- lofactor(train_data[,-13], k=5)
plot(density(outlier.scores))
outliers <- order(outlier.scores, decreasing=T)[1:5] #Taking Top 5 outliers
print(outliers)
#Visualising outliers
n <- nrow(train_data[,-13])
labels <- 1:n
labels[-outliers] <- "."
biplot(prcomp(train_data[,-13]), cex=.8, xlabs=labels)

#Detecting Outliers using interquartile
quantiles <- quantile(train_data$ApplicantIncome, probs = c(.25, .75))
range <- 1.5 * IQR(train_data$ApplicantIncome)
normal_ApplicantIncome <- subset(train_data$ApplicantIncome,
                                 train_data$ApplicantIncome > (quantiles[1] - range) & train_data$ApplicantIncome < (quantiles[2] + range))
#After deleting outliers 50 rows deleted
   ################### Deleting Outliers is Baaad Habit ##############################

#Detecting and deleting Outlier using outlierKD function
outlierKD <- function(dt, var) {
  var_name <- eval(substitute(var),eval(dt))
  tot <- sum(!is.na(var_name))
  na1 <- sum(is.na(var_name))
  m1 <- mean(var_name, na.rm = T)
  par(mfrow=c(2, 2), oma=c(0,0,3,0))
  boxplot(var_name, main="With outliers")
  hist(var_name, main="With outliers", xlab=NA, ylab=NA)
  outlier <- boxplot.stats(var_name)$out
  mo <- mean(outlier)
  var_name <- ifelse(var_name %in% outlier, NA, var_name)
  boxplot(var_name, main="Without outliers")
  hist(var_name, main="Without outliers", xlab=NA, ylab=NA)
  title("Outlier Check", outer=TRUE)
  na2 <- sum(is.na(var_name))
  message("Outliers identified: ", na2 - na1, " from ", tot, " observations")
  message("Proportion (%) of outliers: ", (na2 - na1) / tot*100)
  message("Mean of the outliers: ", mo)
  m2 <- mean(var_name, na.rm = T)
  message("Mean without removing outliers: ", m1)
  message("Mean if we remove outliers: ", m2)
  response <- readline(prompt="Do you want to remove outliers and to replace with NA? [yes/no]: ")
  if(response == "y" | response == "yes"){
    dt[as.character(substitute(var))] <- invisible(var_name)
    assign(as.character(as.list(match.call())$dt), dt, envir = .GlobalEnv)
    message("Outliers successfully removed", "\n")
    return(invisible(dt))
  } else{
    message("Nothing changed", "\n")
    return(invisible(var_name))
  }
}

outlierKD(train_data, ApplicantIncome)

#Imputing Missing Values for training set
library(mice)
imputed_data <- mice(train_data[,c(7,8,9,10)], m=5, maxit = 50, method = 'rf', seed = 500)
train_data[,c(7,8,9,10)] <- complete(imputed_data, 2)
   ######################## Doesn't Work :-( ################################

#imputing Na values using Amelia
#install.packages("Amelia")
library(Amelia)
amelia_fit <- amelia(train_data[,c(7,8,9)], m=5, parallel = "multicore")
train_data[,c(7,8,9)] <- amelia_fit$imputations[[2]]
   ########################## Doesn't Work :-( ##############################
   
#imputing missing values with missForest
#install.packages("missForest")
library(missForest)
imp <- missForest(train_data[,c(7,8,9,10)])
train_data[,c(7,8,9,10)] <- imp$ximp   
   ######################### Works Fine :-) ################################

#Splitting train data in to two datasets for validation
library(caTools)
set.seed(123)
split <- sample.split(train_data$Loan_Status, SplitRatio = 0.80)
trainset <- subset(train_data, split == TRUE)
testset <- subset(train_data, split == FALSE)


#Model building

#Applying kernel PCA for dimensinality reduction
library(kernlab)
kpca = kpca(~., data = trainset, kernel = 'rbfdot', features = 10)
training_set_pca = as.data.frame(predict(kpca, trainset))
training_set_pca$Loan_Status = trainset$Loan_Status
test_set_pca = as.data.frame(predict(kpca, testset))
test_set_pca$Loan_Status = testset$Loan_Status

#Fitting training data to KNN
library(class)
knn_classifier <- knn(train = trainset[,-13], test = testset[,-13], cl = trainset$Loan_Status, k=5, prob = TRUE)

#Fitting training data to SVM
library(e1071)
svm_classifier = svm(formula = Loan_Status ~ .,
                     data = trainset,
                     type = 'C-classification',
                     kernel = 'sigmoid')

#Fitting training data to naive bayes
library(e1071)
nb_classifier = naiveBayes(x = trainset[,-13],
                           y = trainset$Loan_Status)

#Fitting training data to decision tree
library(rpart)
dt_classifier <- rpart(formula = Loan_Status ~ ., data = train_data)

#Fitting training data to Random Forest
library(randomForest)
set.seed(123)
rf_classifier <- randomForest(x = trainset[,-13], y = trainset$Loan_Status, ntree = 500)

#Fitting training data to XGBoost Model
library(xgboost)
xg_classifier <- xgboost(data = as.matrix(trainset[,-13]), label = trainset$Loan_Status, nrounds = 10)

#Fitting training data to Gradient boost model
library(gbm)
trainset <- trainset[,-14]
gb_classifier <- gbm(Loan_Status ~ ., data = trainset,distribution = "gaussian",n.trees = 10000, interaction.depth = 4, shrinkage = 0.01)

#cross validation
library(caret)
train_control<- trainControl(method="cv", number=10, savePredictions = TRUE)
model<- train(Loan_Status~., data=trainset, trControl=train_control, method="rpart")
print(model) #here we can see the accuracy

#Predicting the test results
svm_pred <- predict(svm_classifier, newdata = testset[,-13])
nb_pred <-  predict(nb_classifier, newdata = testset[,-13])
dt_pred <-  predict(dt_classifier, newdata = testset[,-13], type='class')
rf_pred <-  predict(rf_classifier, newdata = testset[,-13])
xg_pred <-  predict(xg_classifier, newdata = as.matrix(testset[,-13]))
xg_pred <- (xg_pred >= 0.5)
n.trees = seq(from=100 ,to=10000, by=100)
gb_pred <- predict(gb_classifier, newdata = testset[,-13], n.trees = n.trees)
kf_pred <- predict(model, newdata = testset[,-13])

#confusion matrix                                                           normal/pca
cm_knn <- table(testset$Loan_Status,knn_classifier)     #98/103 correct predictions 24/19 incorrect predictions
cm_svm <- table(testset$Loan_Status, svm_pred)          #99/101/103 correct predictions 24/21/19 incorrect predictions
cm_nb <- table(testset$Loan_Status, nb_pred)            #103/101 correct predictions 19/21 incorrect predictions
cm_dt <- table(testset$Loan_Status, dt_pred)            #101/106 correct predictions 21/16 incorrect predictions
cm_rf <- table(testset$Loan_Status, rf_pred)            #98/108 correct predictions 24/14 incorrect predictions
cm_xg <- table(testset$Loan_Status, xg_pred)            #84/84 correct predictions 38/38 incorrect predictions
cm_gb <- table(testset$Loan_Status, gb_pred)
cm_kf <- table(testset$Loan_Status, kf_pred) #101/98 correct predictions 21/24 incorrect predictions
