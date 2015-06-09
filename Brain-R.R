library(glmnet)
library(caret)

#####
load('Brain.RData')
x = Xtrain
y = ytrain
zeros = apply(x, 2, function(x) all(x ==0))
x = x[,-zeros]

##### Subsetting data for K-fold CV
Kfold = function(nobs, k){
  index = 1:nobs
  subsets = vector('list', length = k)
  for(i in 1:k){
    subset = sample(index, nobs/k)
    subsets[[i]] = subset
    index = index[which(!(index %in% subset))]
  }
  return(subsets)
}

k = Kfold(1600, 5)
c1 = c(k[[1]],k[[2]],k[[3]],k[[4]])
c2 = c(k[[5]],k[[1]],k[[2]],k[[3]])
c3 = c(k[[4]],k[[5]],k[[1]],k[[2]])
c4 = c(k[[3]],k[[4]],k[[5]],k[[1]])
c5 = c(k[[2]],k[[3]],k[[4]],k[[5]])

##### Lasso Regression
######################
grid = 10^seq(10,-2, length = 100)

###glmnet fit and CV to choose lambda
glm.fit = glmnet(x[c5,],y[c5], lambda = grid)
glm.cv = cv.glmnet(x[c5,],y[c5])

### how many coefficients used
sum(coef(glm.cv) != 0)
lam.min = glm.cv$lambda.min
print(lam.min)

### Predictions using min lambda 
pred = predict(glm.fit, newx =x[k[[1]],] , s = (lam.min-.02))
mean((y[k[[1]]] - pred)^2)
## mse for valset (k 1 to 5 respetively) = 0.660623, .6448597, 
## 0.8187797, .7580421, 1.191556

### Running model on full training set with parameters from CV
############################################################
glm.fit = glmnet(Xtrain,ytrain, lambda = grid)
set.seed(1)
glm.cv = cv.glmnet(Xtrain,ytrain, alpha = 1)

### how many coefficients used
sum(coef(glm.cv) != 0)
which(coef(glm.cv) != 0)
lam.min = glm.cv$lambda.min
print(lam.min)

### Predictions using min lambda 
pred = predict(glm.fit, newx =Xtest, s = (lam.min- .02))
#### Submission
submission = data.frame(id = 1:150, y = rep(0,150))
submission$y = pred
write.table(submission, file = "FinalSub.csv", col.names= T,
            row.names = F, sep=",")