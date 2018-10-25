library(data.table)
library(psych)
library(dplyr)
library(dlookr)
library(e1071)
library(moments)
library(ggplot2)

df = read.csv(file.choose(),stringsAsFactors = T)
str(df)
attach(df)
summary(df)

describe(df)
y = Flare_Up
cbind(freq=table(y), percentage=prop.table(table(y))*100)
col_names <-c("Flare_Up","Demo1","Demo5","DisHis1","DisHis2","DisHis3","DisHis4",
              "DisHis5","DisHis7","LungFun19","Dis1Treat","Dis4Treat","Dis5Treat",
              "Dis6Treat","Dis7")
new = setdiff(colnames(df),col_names)
cat = df[,col_names]
str(cat)
con = df[,new]
f <- apply(cat, MARGIN = 2, FUN = function(x) as.factor(x))
levels(cat)
tot = cbind.data.frame(f,con)
str(tot)
levels(tot$Flare_Up)
library(plyr)

cols_edit<- c(1:15)
for(i in 1:ncol(tot[,cols_edit])) {
  tot[,cols_edit][,i] <- as.factor(mapvalues(tot[,cols_edit][,i],  
                                                from =c("1","2"),to=c("0","1")))
}

skew = apply(con,2,skewness)
x = cbind.data.frame(skew)
print(x>1)

v = c("DisHis1Times","DisHis2Times","DisHis3Times","DisHis6","LungFun8","LungFun14",
      "LungFun17","LungFun18","Dis1","Dis2","Dis2Times","Dis3","Dis3Times",
       "Dis4","Dis5","Dis6","SmokHis1","SmokHis2","SmokHis4")
l = setdiff(colnames(tot),v)
s = tot[,v]
ns = tot[,l]
c = apply(s,MARGIN = 2,FUN = function(x) sqrt(x))
skew1 = apply(c,2,skewness)  


#ag = c("DisHis1Times","DisHis2Times","DisHis3Times","DisHis6","LungFun18","Dis1","Dis2Times",
 #    "Dis3Times","Dis4")
#b = setdiff(colnames(tot),ag)
#c = tot[,ag]
#d = tot[,b]
#e = apply(c,MARGIN = 2,FUN = function(x) sqrt(x))
#rm(e)
#skew2 = apply(ag,2,skewness)          

pairs.panels(df[2:10],pch = ".",gap = 0)
pairs.panels(df[11:21],pch = ".",gap = 0)
pairs.panels(df[22:32],pch = ".",gap = 0)
pairs.panels(df[33:44],pch = ".",gap = 0)
pairs.panels(df[45:55],pch = ".",gap = 0)
pairs.panels(df[56:62],pch = ".",gap = 0)


set.seed(1234)
ind <- sample(2, nrow(tot), replace = T, prob = c(0.8, 0.2))
train <- tot[ind==1,]
levels(train$Flare_Up)
test <- tot[ind==2,]

m = glm(Flare_Up ~ ., family = 'binomial',data = train)
summary(m)
anova(m)
m1 = glm(Flare_Up ~ Demo2+LungFun17+Dis2Times+LungFun14
         +ResQues1a+DisHis2Times,family = 'binomial',data = train)
summary(m1)
library(Boruta)
bor = Boruta(Flare_Up ~ .,data = na.omit(tot),doTrace = 2)

library(earth)
e = earth(Flare_Up ~ .,data = train)
ev = evimp(e)
plot(ev)

library(devtools)
install_github("riv","tomasgreif")
install_github("woe","tomasgreif")
library(woe)
library(riv)
iv_df <- iv.mult(train, y= "Flare_Up", summary=TRUE, verbose=TRUE)
iv <- iv.mult(train, y="Flare_Up", summary=FALSE, verbose=TRUE)
iv.plot.summary(iv_df)


p = predict(m1,newdata = test)
head(p)
range(p)
levels(test$Flare_Up)
p = ifelse(p>1.5,1,0)
head(p)
tab1 <- table(p,test$Flare_Up)
tab1
correct = mean(p == test$Flare_Up)
correct
error = 1 - correct
error

library(rcompanion)
chisq.test(cat[,1],cat[,2])
chisq.test(cat[,3],cat[,4])
chisq.test(cat[,5],cat[,6])
chisq.test(cat[,7],cat[,8])
chisq.test(cat[,9],cat[,10])
chisq.test(cat[,11],cat[,12])
chisq.test(cat[,13],cat[,14])
chisq.test(cat[,14],cat[,15])


#goodness of fit test
with(m1,pchisq(null.deviance - deviance,df.null - df.residual,lower.tail = F))


library(party)
tree = ctree(Flare_Up ~ Demo2+LungFun17+Dis2Times+LungFun14+ResQues1a+DisHis2Times, data = train,
             controls = ctree_control(mincriterion = 0.9,minsplit = 400))
plot(tree,type = "simple")

pred_tree = predict(tree,test)
confusionMatrix(pred_tree,test$Flare_Up)


#random forest
library(randomForest)
rf1  = randomForest(Flare_Up ~ Demo2+LungFun17+Dis2Times+LungFun14+ResQues1a+DisHis2Times,data = train) 
print(rf1)
prf = predict(rf1,test)
confusionMatrix(prf,test$Flare_Up)
plot(rf1)

#tuning random forest model
t <- tuneRF(train[, -62], train[, 62], stepFactor = 0.5, plot = TRUE, 
            ntreeTry = 200, trace = TRUE, improve = 0.05)

rf  = randomForest(Flare_Up~Demo2+LungFun17+Dis2Times+LungFun14+ResQues1a+DisHis2Times,data = train,
                   mtry = 40,ntree = 200,importance = T,proximity =T) 

print(rf)

prt = predict(rf,test)
confusionMatrix(prt,test$Flare_Up)


#variable importance
varImpPlot(rf, sort=T, n.var = 5, main = 'Top 10 Feature Importance')
