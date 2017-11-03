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
setwd(wd)

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
confusionMatrix(data = pc.LR.out, 
                reference = df.train$rich
                )

                ## Accuracy of ~80%. This looks promising! 
                ##   -- In practice, dimensionality reduction is hard.
                ##      We may have got lucky here.






## What if we want to try again, but keep it in original feature space (not PCs)? 
## Lets try a univariate filter (Pearson correlation)

# correlate each variable with the outcome
correlations <- sapply(names(df.train)[-1], 
                       function(i) cor(as.numeric(df.train[,i]), as.numeric(df.train$rich))
                       )

                ## what happened?

summary(correlations)

                ## some modest correlations.
                ## lets only keep |z| > 0.2?

fs1 <- names(correlations)[(correlations > 0.2) | (correlations < -0.2)]

df.fs1 <- dplyr::select(df.train, one_of(c("rich", fs1)))

## Try a logistic regression but only using 
## predictors that have good correlations

fs1.lr <- glm(rich ~ ., family = "binomial", data = df.fs1)

                # the model fit!
                # how did it do?

confusionMatrix(data = ifelse(fs1.lr$fitted.values < 0.5, "no", "yes"),
                reference = df.fs1$rich
                )

                # ~80%, about the same as the PC reduction. 


## Can you think of problems with this analysis?
#     performance measures were very high- why?




# Model Validation -- how do models perform on unseen data?

# Lets try testing our out models on the second half of the class!

# Read in the second half of the class data
df.test      <- read.csv(file.path(wd, "input/class_testing_data.csv"),
                         as.is=TRUE
                         )

df.test$rich <- as.factor(df.test$rich)


## First we will test our PCA approach

# Apply the learned PCA matrix to the test data
df.test.PC <- predict(pca1, 
                      newdata = df.test[ , -1]
                      )[,1:11] %>% 
                          as.data.frame

# Predict `rich` using the PC logistic regression
test.PC.out <- predict(pc.LR, newdata = df.test.PC, type = "response")

# Threshold the predictions
test.PC.out <- ifelse(test.PC.out < 0.5, "no", "yes")

# Create a Confusion Matrix
confusionMatrix(data = test.PC.out, reference = df.test$rich)

# The accuracy is much much lower.


## Next we will test our simple feature selection approach

# Make predictions only using predictors that had good correlations in training data
fs1.test <- predict(fs1.lr, 
                    newdata = df.test[fs1], 
                    type = "response"
                    )

# Threshold the predictions
test.fs1.out <- ifelse(fs1.test < 0.5, "no", "yes")

# how did it do?
confusionMatrix(data = test.fs1.out,
                reference = df.test$rich
                )


# Can you think of reasons why?

## Another approach is to use an actual model to rank features,
##    using some of the data




##### This section is beyond intro #####


### ##################
### Comparing Learners
### ##################


## First we set up cross-validation procedures that we want
##   We will give these instructions to the train command later.

cvCtrl <- trainControl(method = "cv", 
                       number = 10, 
                       classProbs = TRUE
                       )   # see help for other CV


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

# LDA learner
set.seed(1)

mod1 <- train(x = as.matrix(df.train[,-1]),
              y = as.factor(df.train$rich),
              method = "lda",
              trControl = cvCtrl
              )

                      ## inspect structure of model we built
                      ## (broadly, don't worry too much)

## remember getTrainPerf?
getTrainPerf(mod1)

                      # not great! this is the
                      # average cross-validated performance

## Once you have the framework set up, changing algorithm is trivial

## SVM learner

set.seed(1)

mod2 <- train(x = as.matrix(df.train[, -1]),
              y = as.factor(df.train$rich),
              method = "svmLinear2",
              trControl = cvCtrl
)

getTrainPerf(mod2)
# SVM did about the same as LDA

## KNN learner

set.seed(1)

mod3 <- train(x = as.matrix(df.train[, -1]),
              y = as.factor(df.train$rich),
              method = "knn",
              trControl = cvCtrl
)

getTrainPerf(mod3)

## Random Forest learner

set.seed(1)

mod4 <- train(x = as.matrix(df.train[, -1]),
              y = as.factor(df.train$rich),
              method = "rf",
              trControl = cvCtrl
)

getTrainPerf(mod4) # not great! 


## Neural Network learner

set.seed(1)

mod5 <- train(x = as.matrix(df.train[, -1]),
              y = as.factor(df.train$rich),
              method = "nnet",
              trControl = cvCtrl
)

getTrainPerf(mod5)

## Naive Bayes learner

set.seed(1)

mod6 <- train(x = as.matrix(df.train[, -1]),
              y = as.factor(df.train$rich),
              method = "bayesglm",
              trControl = cvCtrl
)

getTrainPerf(mod6)


### ########################
### Out of Sample Prediction
### ########################

## Once you have reached this point, you are ready to test your best algorithm
##    on some unseen data to see how well it performs


## The caret model structure can be used to predict new outcomes
##     there is a function called predict that you use to make predictions with a model

mod2$method
mod2.out <- predict(mod2, newdata = df.test[ -1])
confusionMatrix(data = mod2.out, reference = df.test$rich)

mod3$method
mod3.out <- predict(mod3, newdata = df.test[ -1])
confusionMatrix(data = mod3.out, reference = df.test$rich)

mod4$method
mod4.out <- predict(mod4, newdata = df.test[ -1])
confusionMatrix(data = mod4.out, reference = df.test$rich)

mod5$method
mod5.out <- predict(mod5, newdata = df.test[ -1])
confusionMatrix(data = mod5.out, reference = df.test$rich)

mod6$method
mod6.out <- predict(mod6, newdata = df.test[ -1])
confusionMatrix(data = mod6.out, reference = df.test$rich)

## almost all of these models had serious overfitting issues!







### #################################
### Moving beyond basic `caret` usage
### #################################

## https://topepo.github.io/caret/

## Sometimes you need additional packages to run certain non-standard models

## This is a little obscure, but here we can check what models are available
#    if we wanted to do classification/or dual use

t <- getModelInfo()
m <- list()
for (i in names(t)){
  if (t[[i]]$type != "Regression"){
    m <- c(m, t[i])
  }
}
names(m)[1:5]
names(m)



## You can also change the cross-validation framework FYI
## here is an example of leave-one-out CV

areyoupatient <- FALSE

if (areyoupatient) {
  modLOO <- train(x= as.matrix(df.train[, -1]),
                  y = as.factor(df.train$rich),
                  method = "svmLinear",
                  trControl = trainControl(method="LOOCV")
                  )
                                  # it is typically very slow. don't run
                                  # it. there are as many folds /
                                  # re-estimations as there are
                                  # observations
  }


### ######################
### Tuning Hyperparameters
### ######################

## We mentioned briefly in the workshop that some algorithms require tuning
#    Best way to do this is to pre-specify a grid of all the parameter
#    combinations that you want to try, and choose the best through CV

## Hyperparameter grid (aka tuning grid) can be set up easily
## Example with random forests
##   - visit caret docs for detail:
##     http://topepo.github.io/caret/modelList.html

## The "catch" is that tuning varies from learning algorithm to learning
## algorithm.

## Tying it all together

set.seed(123)

modDef <- train(x = as.matrix(df.train[, -1]),
                y = as.factor(df.train$rich),
                method = "rf",
                trControl = cvCtrl
)
                      # default hyperparameter for number of
                      # variables to include in individual
                      # model
                      #
                      # default is 3 different values

print(modDef)

getTrainPerf(modDef)


set.seed(123)

modNoGrid <- train(x = as.matrix(df.train[, -1]),
                   y = as.factor(df.train$rich),
                   method = "rf",
                   tuneLength = 10,
                   trControl = cvCtrl
)
                      #
                      # same behavior, but for 10 values

print(modNoGrid)
ggplot(modNoGrid)
getTrainPerf(modNoGrid)

## Now, we explicitly explore parts of the hyperparameter space

rfGrid <- expand.grid(mtry = seq(3, 21, by = 3))

set.seed(123)

modGrid <- train(x = as.matrix(df.train[, -1]),
                 y = as.factor(df.train$rich),
                 method = "rf",
                 tuneGrid = rfGrid,
                 trControl = cvCtrl
)

print(modGrid)

getTrainPerf(modGrid)


# try smaller values of mtry, but sampling more closely?
rfGrid2 <- expand.grid(mtry = seq(3, 10, by = 1))

set.seed(123)

modGrid2 <- train(x = as.matrix(df.train[, -1]),
                  y = as.factor(df.train$rich),
                  method = "rf",
                  tuneGrid = rfGrid2,
                  trControl = cvCtrl
)

print(modGrid2)

getTrainPerf(modGrid2)


## Out of Sample Comparison of Fit for Various Tuning Runs

modDef$bestTune
modDef.out <- predict(modDef, newdata = df.test[ -1])
confusionMatrix(data = modDef.out, reference = df.test$rich)

modNoGrid$bestTune
modNoGrid.out <- predict(modNoGrid, newdata = df.test[ -1])
confusionMatrix(data = modNoGrid.out, reference = df.test$rich)

modGrid$bestTune
modGrid.out <- predict(modGrid, newdata = df.test[ -1])
confusionMatrix(data = modGrid.out, reference = df.test$rich)

modGrid2$bestTune
modGrid2.out <- predict(modGrid2, newdata = df.test[ -1])
confusionMatrix(data = modGrid2.out, reference = df.test$rich)

## parameter tuning seemed to help here because the defaults were inappropriate
## for these data it takes time/experience to get used to tuning algorithms
## overall, biggest issue in this instance is probably inappropriate algorithm selection

################################################################################
################################################################################
################################################################################











