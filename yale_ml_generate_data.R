### Introduction to Machine Learning
###  Yale University
###  Author: Adam Chekroud

#### script to generate the data set


library(caret); library(dplyr)

## Generate data
set.seed(1)
df <- twoClassSim(n = 100, 
                  linearVars = 5, noiseVars = 5,
                  corrVars = 5, corrValue = 0.5)[,3:21]
df <- df %>% 
  mutate(rich = factor(Class, labels = c("yes", "no") ) ) %>% 
  select(rich, Linear1:Corr5, -Class)
names(df)[2:6] <- c("age", "sex", "income", "race", "employment")
names(df)[7:11] <- c("diabetes", "hypertension", "cancer","red", "blue")
names(df)[12:16] <- c("depression", "infection", "green", "yellow", "black")
df <- df %>% select(-Corr3, -Corr4, -Corr5)
df$sex <- ifelse(df$sex < median(df$sex), "0", "1")
df$income <- -1 * df$income
write.csv(df, "class_data.csv")
#glm(rich ~ . , data = df, family = "binomial") %>% summary()