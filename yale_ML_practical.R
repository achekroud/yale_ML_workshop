### Introduction to Machine Learning
###  Yale University
###  Author: Adam Chekroud

##############
##############
######### Set working directory to the repository you downloaded

setwd("~/")

##############
##############

# Load libraries
# use these commands to install packages on your local machine
# install.packages('caret')
# install.packages('dplyr')
# install.packages('ggplot2')
# install.packages('e1071')
library(ggplot2); library(caret); library(dplyr); library(e1071)

## Read in class data
#     Data contains sociodemographics, medical history, and color prefs,
#     for 250 people bunch of people surveyed last week.
#     The outcome of interest here is whether they self-identify as rich

df <- read.csv("~/class_data.csv", as.is=TRUE)
df$X <- NULL
df$rich <- as.factor(df$rich)


## View data in Rstudio
# View(df)

## View top of data frame in console
head(df[,1:7])

## Any missing data?
complete.cases(df) %>% table()

## Do higher income ppl identify as rich?
ggplot(data = df) + 
  geom_boxplot(aes(y = income, x = rich)) + coord_flip()

## Typical approach: logistic regression
lr1 <- glm(rich ~ ., family = "binomial", data = df)
summary(lr1)
# In this case, we would probably stop here!

# extract fitted values, see how well they predict feeling rich
lr1.out <- fitted(lr1)

# looks pretty good, considerable separation according to our predicted values
ggplot(data = cbind(df, lr1.out)) + 
  geom_boxplot(aes(y = lr1.out, x = rich), width = 0.2 ) + 
  coord_flip()

# Threshold the predictions, since we have a binary outcome
lr1.bin.out <- ifelse(lr1.out < 0.5, "no", "yes")

# Confusion Matrices are really easy! Use this function!
confusionMatrix(data = lr1.bin.out, reference = df$rich)

# 78% Accuracy! Not bad! 


# but what if we had more than 15 variables, and instead had 30?
rand1 <- df[,2:16] + matrix(nrow = 100, ncol = 15, rnorm(15*100))
rand2 <- df[,2:16] + matrix(nrow = 100, ncol = 15, rnorm(15*100))
df2 <- cbind(df, rand1, rand2)
names(df2)[17:46] <- c(paste0(names(df[2:16]), "_2"), paste0(names(df[2:16]), "_3"))


# run another logistic regression, with 30 predictors and 100 subjects
lr2 <- glm(rich ~ ., family = "binomial", data = df2)
# doesn't work! 

# Maybe we should have tried to do feature selection. 
# Lets try Principle Components Analysis
df.pc <- prcomp(df2[,-1])$x %>% as.data.frame()
# Take top 10 components (guess)
df.pc <- df.pc[,1:11]
# Put rich back in
df.pc$rich <- df2$rich

# Try the logistic regression again?
pc.LR <- glm(rich ~ ., family = "binomial", data = df.pc)
summary(pc.LR)
# Yay! Reducing the dimensionality of the problem (using PCA) allowed us to fit the model 

# Extract the predictions
pc.LR.out <- pc.LR$fitted.values
# Threshold them
pc.LR.out <- ifelse(pc.LR.out < 0.5, "no", "yes")

# Confusion Matrices are really easy! Use this function!
confusionMatrix(data = pc.LR.out, reference = df.pc$rich)

# Accuracy of ~72%. This is less than we had in the first place! Dimensionality reduction is hard. 


# What if we want to try again, but keep it in original feature space (not PCs)? 
# Lets try a univariate filter (Pearson correlation)

# correlate each variable with the outcome
correlations <- sapply(names(df2)[2:31], function(i) cor(as.numeric(df2[,i]), as.numeric(df2$rich)))
# what happened?
summary(correlations)
# some modest correlations. lets only keep |z| > 0.1?


fs1 <- names(correlations)[(correlations > 0.1) | (correlations < -1.5)]
df.fs1 <- dplyr::select(df2, one_of(c("rich", fs1)))

# stronger predictors, logistic regression
fs1.lr <- glm(rich ~ ., family = "binomial", data = df.fs1)
# how was it?
confusionMatrix(data = ifelse(fs1.lr$fitted.values < 0.5, "no", "yes"), reference = df.fs1$rich)
# ~80%, better than the PC reduction, but not much better than the original model. 


## Can you think of problems with this analysis?
#     performance measures were very high- why?

##### This section is beyond intro #####
# Code to do cross-validated univariate feature selection using ANOVA
# Really slow, computationally unstable if fitted model is complicated (only LDA ran)
# Code for RFE is similar but worse

### TODO
# mySBF <- caretSBF
# mySBF$filter <- function(score, x, y) { score <= 0.00001 }

# sbf1 <- sbf(x = as.matrix(df[,2:16]), y = as.factor(df$sex),
#             method = "lda",
#             trControl = trainControl(method = "none", 
#                                      classProbs = TRUE),
#             sbfControl = sbfControl(functions = mySBF,
#                                    method = "cv"))
# 94% average test fold performance with small SD
# almost all variables were kept
# what might we do to change this? new score, multivariate filter, RFE, different scoring function
### END TODO


## Another approach is to use an actual model to rank features,
##    using just a subset of the data

# There is a function called createDataPartition that creates random 
#   splits of the data, and balances class outcomes between the two splits
set.seed(1)    # this means that we can get the same random split next time
inSubset <- createDataPartition(df2$rich, p=0.75, list=FALSE)
df.sub   <- df2[inSubset,]
df.rest  <- df2[-inSubset,]

# Fit a model in the subset
mod.sub <- train(x= as.matrix(df.sub[,2:31]), y = as.factor(df.sub$rich),
                        method = "svmLinear")
# How did it do? getTrainPerf will go and get the performance metrics quickly for you
getTrainPerf(mod.sub)
# can also print the model output
print(mod.sub)

# Extract variable importance from the model
plot(varImp(mod.sub))                             # all of the predictors!
plot(varImp(mod.sub), top = 10)                   # Just top 10, scaled
plot(varImp(mod.sub, scale = FALSE), top = 10)    # Can have raw importance

# Can extract raw coefficients for the final model
coef(mod.sub$finalModel) %>% head()
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
mod1 <- train(x = as.matrix(df.sub[,2:31]),
              y = as.factor(df.sub$rich),
              method = "lda",
              trControl = cvCtrl)

## inspect structure of model we built (broadly, don't worry too much)

## remember getTrainPerf?
getTrainPerf(mod1) #not great! this is the average cross-validated performance



## Once you have the framework set up, changing algorithm is trivial
#   Here is an SVM
set.seed(1)
mod2 <- train(x= as.matrix(df.sub[,2:31]),
              y = as.factor(df.sub$rich),
              method = "svmLinear2",
              trControl = cvCtrl)
getTrainPerf(mod2) # SVM did about the same as LDA

#   Here is k-NN (not run), as an example of how you can try other models
# set.seed(2)
# mod3 <- train(x= as.matrix(df.sub[,2:31]),
#               y = as.factor(df.sub$rich),
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
# mod5 <- train(x= as.matrix(df.sub[,2:31]),
#               y = as.factor(df.sub$rich),
#               method = "svmLinear",
#               trControl = trainControl(method="LOOCV"))
# # it is typically very slow. don't run it.


## Once you have reached this point, you are ready to test your best
##    algorithm on some unseen data to see how well it performs


## The caret model structure can be used to predict new outcomes 
#     there is a function called predict that you use to make predictions with a model
mod2.out <- predict(mod2, newdata = df.rest[,2:31])

confusionMatrix(data = mod2.out, reference = df.rest$rich)
# about 70% accuracy on the 24 people we left out







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
mod6 <- train(x = as.matrix(df.sub[,2:31]),
              y = as.factor(df.sub$rich),
              method = "svmRadial",
              tuneGrid = svmGrid,
              trControl = cvCtrl)
getTrainPerf(mod6) 

print(mod6) # Inspect the model summary
# parameter tuning seemed to help here because the defaults were inappropriate for these data
# it takes time/experience to get used to tuning algorithms.

# what do you think about this model? good? real? why?




# test the fancier model on the left out participants
mod6.out <- predict(mod6, newdata = df.rest[,2:31])

# see how well it did
confusionMatrix(data = mod6.out, reference = df.rest$rich)

# seems like it did even better, even tho we may have concerns about overfitting










