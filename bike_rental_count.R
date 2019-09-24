rm(list=ls())

# set working directory
setwd("F:/R_Programming/Edwisor")
getwd()

##############################################################

# loading Libraries
x = c("tidyr", "ggplot2", "corrgram", "usdm", "caret", "DMwR", "rpart", "randomForest",'xgboost')

# tidyr - drop_na
# ggplot2 - for visulization, boxplot, scatterplot
# corrgram - correlation plot
# usdm - vif
# caret - createDataPartition
# DMwR - regr.eval
# rpart - decision tree
# randomForest - random forest
# xgboost - xgboost

# load Packages
lapply(x, require, character.only = TRUE)
rm(x)

#############################################################

# loading dataset
df = read.csv("day.csv", header = T, na.strings = c(" ", "", "NA"))

######################
# Exploring Datasets
######################

# Structure of data
str(df)

# Summary of data
summary(df)

# Viewing the data
head(df,5)

#####################################
# EDA, Missing value and Outlier analysis
#####################################

# Changing the data types of variables
df$dteday = as.Date(as.character(df$dteday))

catnames=c("season","yr","mnth","holiday","weekday","workingday","weathersit")
for(i in catnames){
  print(i)
  df[,i]=as.factor(df[,i])
}

# Checking Missing data
apply(df, 2, function(x) {sum(is.na(x))})
# No missing values are present in the given data set.

#Outlier Analysis
num_index = sapply(df, is.numeric)
numeric_data = df[,num_index]
num_cnames = colnames(numeric_data)

for (i in 1:length(num_cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (num_cnames[i]), x = "cnt"), data = subset(df))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=num_cnames[i],x="cnt")+
           ggtitle(paste("Box plot of count for",num_cnames[i])))
}

# ## Plotting plots together
gridExtra::grid.arrange(gn1,gn2,ncol=2)
gridExtra::grid.arrange(gn4,gn5,ncol=2)
gridExtra::grid.arrange(gn6,gn3,ncol=2)

# continous variables hum, windspeed and casual includes outliers
# we do not consider casual variable for outlier removal bcz this is not predictor variable

outlier_var=c("hum","windspeed")

#Replace all outliers with NA

for(i in outlier_var){
  val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
  print(length(val))
  df[,i][df[,i] %in% val] = NA
}

# Checking Missing data - after outlier
apply(df, 2, function(x) {sum(is.na(x))})

# hum includes 2 outliers and windspeed includes 13 outliers, so drop them
df = drop_na(df)

# Make a copy
df_after_outlier = df

############### Visualization ########################
# Scatter plot between temp and cnt
ggplot(data = df, aes_string(x = df$temp, y = df$cnt))+ 
  geom_point()

# Scatter plot between atemp and cnt
ggplot(data = df, aes_string(x = df$atemp, y = df$cnt))+ 
  geom_point()

# Scatter plot between hum and cnt
ggplot(data = df, aes_string(x = df$hum, y = df$cnt))+ 
  geom_point()

# Scatter plot between windspeed and cnt
ggplot(data = df, aes_string(x = df$windspeed, y = df$cnt))+ 
  geom_point()

# Scatter plot between season and cnt
ggplot(data = df, aes_string(x = df$season, y = df$cnt))+ 
  geom_point()

# Scatter plot between month and cnt
ggplot(data = df, aes_string(x = df$mnth, y = df$cnt))+ 
  geom_point()

# Scatter plot between weekday and cnt
ggplot(data = df, aes_string(x = df$weekday, y = df$cnt))+ 
  geom_point()

##################### Feature Selection and Scaling #######################
# generate correlation plot between numeric variables

numeric_index=sapply(df, is.numeric)
corrgram(df[,numeric_index], order=F, upper.panel=panel.pie, 
         text.panel=panel.txt, main="Correlation plot")

# check VIF
vif(df[,10:15])
# if vif is greater than 10 then variable is not suitable/multicollinerity

# ANOVA test for checking p-values of categorical variables
for (i in catnames) {
  print(i)
  print(summary(aov(df$cnt ~df[,i], df)))
}

#From correlation plot and VIF, Removing variables atemp beacuse it is highly correlated with temp,
#From Anova, Removing weekday, holiday because they don't contribute much to the independent cariable
#Removing Causal and registered becuase that's what we need to predict.
#Removing instant and dteday because they are not useful in generating model.

# remove the variables
df=subset(df,select=-c(instant,dteday,atemp,casual,registered,holiday,weekday))

# Make a copy
df_clean = df

# generate histogram of continous variables
qqnorm(df$temp)
hist(df$temp)

qqnorm(df$hum)
hist(df$hum)

qqnorm(df$windspeed)
hist(df$windspeed)

######################### Model Development ###################

############ Splitting df into train and test ###################
set.seed(101)
split_index = createDataPartition(df$cnt, p = 0.80, list = FALSE) 
train_data = df[split_index,]
test_data = df[-split_index,]


#############  Linear regression Model  #################
lm_model = lm(cnt ~., data=train_data)

# summary of trained model
summary(lm_model)

# prediction on test_data
lm_predictions = predict(lm_model,test_data[,1:8])

regr.eval(test_data[,9],lm_predictions)
#   mae          mse         rmse       mape 
#  5.618664e+02 5.535047e+05 743.979  0.1728075

# compute r^2
rss_lm = sum((lm_predictions - test_data$cnt) ^ 2)
tss_lm = sum((test_data$cnt - mean(test_data$cnt)) ^ 2)
rsq_lm = 1 - rss_lm/tss_lm
#    r^2 - 0.8407258

############## Decision Tree Model ###############
Dt_model = rpart(cnt ~ ., data=train_data, method = "anova")

# summary on trainned model
summary(Dt_model)

#Prediction on test_data
predictions_DT = predict(Dt_model, test_data[,1:8])

regr.eval(test_data[,9], predictions_DT)
#   mae          mse         rmse         mape 
# 7.382928e+02 9.544681e+05 976.968     0.263328

# compute r^2
rss_dt = sum((predictions_DT - test_data$cnt) ^ 2)
tss_dt = sum((test_data$cnt - mean(test_data$cnt)) ^ 2)
rsq_dt = 1 - rss_dt/tss_dt
#    r^2 - 0.7253463

#############  Random forest Model #####################
rf_model = randomForest(cnt ~., data=train_data)

# summary on trained model
summary(rf_model)

# prediction of test_data
rf_predictions = predict(rf_model, test_data[,1:8])

regr.eval(test_data[,9], rf_predictions)
#  mae           mse         rmse         mape 
# 4.944837e+02 4.197740e+05 647.899    0.172765

# compute r^2
rss_rf = sum((rf_predictions - test_data$cnt) ^ 2)
tss_rf = sum((test_data$cnt - mean(test_data$cnt)) ^ 2)
rsq_rf = 1 - rss_rf/tss_rf
#    r^2 - 0.8792076

############  XGBOOST Model ###########################
train_data_matrix = as.matrix(sapply(train_data[-9],as.numeric))
test_data_matrix = as.matrix(sapply(test_data[-9],as.numeric))

xgboost_model = xgboost(data = train_data_matrix,label = train_data$cnt, nrounds = 15,verbose = FALSE)

# summary of trained model
summary(xgboost_model)

# prediction on test_data
xgb_predictions = predict(xgboost_model,test_data_matrix)

regr.eval(test_data[,9], xgb_predictions)
#   mae          mse         rmse      mape 
# 4.848618e+02 4.511432e+05 671.671   0.153581

# compute r^2
rss_xgb = sum((xgb_predictions - test_data$cnt) ^ 2)
tss_xgb = sum((test_data$cnt - mean(test_data$cnt)) ^ 2)
rsq_xgb = 1 - rss_xgb/tss_xgb
#    r^2 - 0.870181


# from above models, it is clear that xgboost is best model
