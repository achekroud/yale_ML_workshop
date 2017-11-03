### Introduction to Machine Learning
###  Yale University
###  Author: Adam Chekroud

#### script to generate the data set

# load necessary libraries
# use these commands to install packages on your local machine
# install.packages('caret')
# install.packages('dplyr')
library(caret); library(dplyr)

## Generate data
set.seed(1)
df <- twoClassSim(n = 100, 
                  linearVars = 5, noiseVars = 10,
                  corrVars = 15, corrValue = 0.8)
df <- df %>% 
  mutate(rich = factor(Class, labels = c("yes", "no") ) ) %>% # convert class variable to yes/no
  select(rich, Linear1:Corr15, -Class) # remove the Class column from the data frame

names(df)[2:6]   <- c("age", "female", "income", "race", "employment")
names(df)[7:9]   <- c("diabetes", "hypertension", "cancer")
names(df)[10:19] <- c("red", "orange", "yellow", "green", "blue", 
                      "indigo", "violet","gold", "silver", "bronze")
names(df)[20:ncol(df)] <- letters[1:15]

df$female <- ifelse(df$female < median(df$female), "0", "1")
df$income <- -1 * df$income
write.csv(df[1:50, ], file.path(getwd(), "class_training_data.csv")) 
write.csv(df[51:100, ], file.path(getwd(), "class_testing_data.csv")) 

# glm(rich ~ . , data = df[1:50,], family = "binomial") %>% summary()


# but what if we had more than 15 variables, and instead had 30?
rand1 <- df[,2:16] + matrix(nrow = 100, ncol = 15, rnorm(15*100))
rand2 <- df[,2:16] + matrix(nrow = 100, ncol = 15, rnorm(15*100))
df2 <- cbind(df, rand1, rand2)
names(df2)[17:46] <- c(paste0(names(df[2:16]), "_2"), paste0(names(df[2:16]), "_3"))

