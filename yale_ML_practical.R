################################################################################
### Machine Learning in R
###
### AUTHOR: Adam Chekroud
################################################################################

### ######
### Set up
### ######

## Set the working directory to where this script lives. Details by machine will
## vary.

wd <- setwd("/Users/adamchekroud/Documents/PhD/PhD_Core/Teaching/yale_ml_workshop")

### ########
### Packages
### ########

## use these commands to install packages on your local machine

## install.packages('caret')
## install.packages('dplyr')
## install.packages('ggplot2')
## install.packages('e1071')

library("ggplot2")
library("caret")
library("dplyr")
library("e1071")

### ############
### Read in Data
### ############

##     Data contains sociodemographics, medical history, and color prefs,
##     for 100 people people surveyed last week.
##     The outcome of interest here is whether they self-identify as rich
##     The class were split into two equal groups, called training and testing

# Read data
df.train      <- read.csv(file.path(wd, "input/class_training_data.csv"),
                          as.is=TRUE
                          )

df.train$rich <- as.factor(df.train$rich)

                    # tell R to treat binary outcome as a
                    # factor


### ###########
### Review Data
### ###########

## View data in Rstudio
## View(df)

## View top of data frame in console
head(df.train[,1:7])

## Any missing data?
complete.cases(df.train) %>% table()

## Do higher income ppl identify as rich?
ggplot(data = df.train) + 
  geom_boxplot(aes(y = income, x = rich)) +
  coord_flip()

### ##############
### Basic Analysis
### ##############

## Typical approach: logistic regression
lr1 <- glm(rich ~ ., family = "binomial", data = df.train)
summary(lr1)

## It didn't work! 
##    The model failed because there were too many variables in
##    the model and it found perfect separation



### ####################################
### Feature Selection/Variable Selection
### ####################################



## Maybe we should have tried to do feature selection. 
## Lets try Principle Components Analysis

pca1  <- prcomp(df.train[,-1])
df.pc <- pca1$x %>% as.data.frame()

## Take top 10 components (guess)
df.pc <- df.pc[,1:11]

## Put rich (outcome) back in
df.pc$rich <- df.train$rich

## Try the logistic regression again?
pc.LR <- glm(rich ~ ., family = "binomial", data = df.pc)
summary(pc.LR)

##  Yay! 
##    Reducing the dimensionality of the problem (using PCA)
##    allowed us to fit the model 

## Lets see how well the model did! 

## Extract the predictions
pc.LR.out <- pc.LR$fitted.values

## Threshold the predictions, since we have a binary outcome
pc.LR.out <- ifelse(pc.LR.out < 0.5, "no", "yes")

## Confusion Matrices are really easy! Use this function!
confusionMatrix(data = pc.LR.out, reference = df.train$rich)

## Accuracy of ~82%. This looks promising! 
##   -- In practice, dimensionality reduction is hard.
##      We may have got lucky here.






# What if we want to try again, but keep it in original feature space (not PCs)? 
# Lets try a univariate filter (Pearson correlation)

# correlate each variable with the outcome
correlations <- sapply(names(df.train)[2:34], 
                       function(i) cor(as.numeric(df.train[,i]), as.numeric(df.train$rich)))
# what happened?
summary(correlations)

# some modest correlations. lets only keep |z| > 0.2?
fs1 <- names(correlations)[(correlations > 0.2) | (correlations < -0.2)]
df.fs1 <- dplyr::select(df.train, one_of(c("rich", fs1)))

# Try a logistic regression but only using predictors that have good correlations
fs1.lr <- glm(rich ~ ., family = "binomial", data = df.fs1)
# the model fit!

# how did it do?
confusionMatrix(data = ifelse(fs1.lr$fitted.values < 0.5, "no", "yes"),
                reference = df.fs1$rich)
# ~80%, about the same as the PC reduction. 


## Can you think of problems with this analysis?
#     performance measures were very high- why?




# Model Validation -- how do models perform on unseen data?

# Lets try testing our out models on the second half of the class!

# Read in the second half of the class data
df.test      <- read.csv(file.path(wd, "class_testing_data.csv"), as.is=TRUE)
df.test$rich <- as.factor(df.test$rich)


## First we will test our PCA approach

# Apply the learned PCA matrix to the test data
df.test.PC <- predict(pca1, newdata = df.test[ , -1])[,1:11] %>% as.data.frame

# Predict `rich` using the PC logistic regression
test.PC.out <- predict(pc.LR, newdata = df.test.PC, type = "response")

# Threshold the predictions
test.PC.out <- ifelse(test.PC.out < 0.5, "no", "yes")

# Create a Confusion Matrix
confusionMatrix(data = test.PC.out, reference = df.test$rich)

# The accuracy is much much lower.


## Next we will test our simple feature selection approach

# Make predictions only using predictors that had good correlations in training data
fs1.test <- predict(fs1.lr, newdata = df.test[fs1], type = "response")

# Threshold the predictions
test.fs1.out <- ifelse(fs1.test < 0.5, "no", "yes")

# how did it do?
confusionMatrix(data = test.fs1.out,
                reference = df.test$rich)

# This performed even worse!

# Can you think of reasons why?





##### This section is beyond intro #####



# Code to do cross-validated univariate feature selection using ANOVA
# Really slow, computationally unstable if fitted model is complicated (only LDA ran)
# Code for RFE is similar but worse


# mySBF <- caretSBF
# mySBF$filter <- function(score, x, y) { score <= 0.00001 }

# sbf1 <- sbf(x = as.matrix(df.train[,2:34]), y = as.factor(df.train$rich),
#             method = "lda",
#             trControl = trainControl(method = "none", 
#                                      classProbs = TRUE),
#             sbfControl = sbfControl(functions = mySBF,
#                                    method = "cv"))
# 94% average test fold performance with small SD
# almost all variables were kept
# what might we do to change this? new score, multivariate filter, RFE, different scoring function


## Another approach is to use an actual model to rank features,
##    using just a subset of the data

# There is a function called createDataPartition that creates random 
#   splits of the data, and balances class outcomes between the two splits

# Fit a model in the subset
svm1 <- train(x= as.matrix(df.train[,2:34]), y = as.factor(df.train$rich),
                        method = "svmLinear")
# How did it do? getTrainPerf will go and get the performance metrics quickly for you
getTrainPerf(svm1)
# can also print the model output
print(svm1)

# Extract variable importance from the model
plot(varImp(svm1))                             # all of the predictors!
plot(varImp(svm1), top = 10)                   # Just top 10, scaled
plot(varImp(svm1, scale = FALSE), top = 10)    # Can have raw importance

# Can extract raw coefficients for the final model
coef(svm1$finalModel) %>% head()
# I would usually rank/analyse these, and then take the names of the best ones for further modeling







#####

##  Next we will learn how to fit models using cross-validation
##      tools that are built in automatically.




## First we set up cross-validation procedures that we want
#   We will give these instructions to the train command later.
cvCtrl <- trainControl(method = "cv", number = 10, classProbs = TRUE)   # see help for other CV


### The train command is the core function in caret
##      it ties everything together in an **extremely** convenient wrapper

## Pseudocode train command (for humans to read)
# 
# modelStructure <-             ### you need to save all the stuff you build!
#     train (                   ### caret::train function
#     predictors as a matrix    ### x-matrix
#     target as a factor        ### dependent variable/target
#     what algorithm are you using?  ### method = "thing"
#     any cross-validation instructions? 
#     ))


# lets see this in practice!
set.seed(1)
mod1 <- train(x = as.matrix(df.train[,2:34]),
              y = as.factor(df.train$rich),
              method = "lda",
              trControl = cvCtrl)

## inspect structure of model we built (broadly, don't worry too much)

## remember getTrainPerf?
getTrainPerf(mod1) #not great! this is the average cross-validated performance



## Once you have the framework set up, changing algorithm is trivial
#   Here is an SVM
set.seed(1)
mod2 <- train(x= as.matrix(df.train[,2:34]),
              y = as.factor(df.train$rich),
              method = "svmLinear2",
              trControl = cvCtrl)
getTrainPerf(mod2) # SVM did about the same as LDA

#   Here is k-NN (not run), as an example of how you can try other models
# set.seed(2)
# mod3 <- train(x= as.matrix(df.train[,2:31]),
#               y = as.factor(df.train$rich),
#               method = "knn",
#               trControl = cvCtrl)
# getTrainPerf(mod3) # kNN actually did much better than other methods

# Sometimes you need additional libraries to run certain non-standard models

##### advanced - see all the available algorithms (over 100) ####
## This is a little obscure, but here we can check what models are available
#    if we wanted to do classification/or dual use
t <- getModelInfo()
m <- list();
for (i in names(t)){
  if (t[[i]]$type != "Regression"){
    m <- c(m, t[i])
  }
}
names(m)[1:5]
#####


## You can also change the cross-validation framework FYI
# here is an example of leave-one-out CV
# mod5 <- train(x= as.matrix(df.train[,2:34]),
#               y = as.factor(df.train$rich),
#               method = "svmLinear",
#               trControl = trainControl(method="LOOCV"))
# # it is typically very slow. don't run it.


## Once you have reached this point, you are ready to test your best
##    algorithm on some unseen data to see how well it performs


## The caret model structure can be used to predict new outcomes 
#     there is a function called predict that you use to make predictions with a model
mod2.out <- predict(mod2, newdata = df.test[,2:34])

# check accuracy on the second half of the class
confusionMatrix(data = mod2.out, reference = df.test$rich)







#### Advanced
## We mentioned briefly in the workshop that some algorithms require tuning
#    Best way to do this is to pre-specify a grid of all the parameter
#    combinations that you want to try, and choose the best through CV

# Hyperparameter grid (aka tuning grid) can be set up easily
# Example with radial SVM 
#   - visit caret docs for detail:
#     http://topepo.github.io/caret/modelList.html

svmGrid <- expand.grid(.sigma = c(1, 0.1, 0.05),
                       .C = c(1.0, 0.5, 0.1, 0.05))

# We just have to pass this tuning grid to the train command (w/ CV)

# Tying it all together
set.seed(123)
mod6 <- train(x = as.matrix(df.train[,2:34]),
              y = as.factor(df.train$rich),
              method = "svmRadial",
              tuneGrid = svmGrid,
              trControl = cvCtrl)
getTrainPerf(mod6) 

print(mod6) # Inspect the model summary
# parameter tuning seemed to help here because the defaults were inappropriate for these data
# it takes time/experience to get used to tuning algorithms.

# what do you think about this model? good? real? why?




# test the fancier model on the left out participants
mod6.out <- predict(mod6, newdata = df.test[,2:34])

# see how well it did
confusionMatrix(data = mod6.out, reference = df.test$rich)











