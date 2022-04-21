# Why reuse a trainControl?
# So you can use the same summaryFunction and tuning parameters for multiple models.
# So you don't have to repeat code when fitting multiple models.
# So you can compare models on the exact same training and test data.

library(caret)
library(ranger)  # for random forest model

# THE DATASET ISN'T LOADED HERE SO THE CODE WON'T WORK

# Create custom indices: myFolds
myFolds <- createFolds(churn_y, k = 5)

# Create reusable trainControl object: myControl
myControl <- trainControl(
  summaryFunction = twoClassSummary,
  classProbs = TRUE,  # IMPORTANT!
  verboseIter = TRUE,
  savePredictions = TRUE,
  index = myFolds
)


# Now that you have a reusable trainControl object called myControl, you can start fitting different predictive models to your churn dataset and evaluate their predictive accuracy.
# You'll start with glmnet, which penalizes linear and logistic regression models on the size and number of coefficients to help prevent overfitting.

# Fit glmnet model: model_glmnet
model_glmnet <- train(
  x = churn_x, 
  y = churn_y,
  metric = "ROC",
  method = "glmnet",
  trControl = myControl
)


# random forest model: combines an ensemble of non-linear decision trees into a highly flexible (and usually quite accurate) model.

# Fit random forest: model_rf
model_rf <- train(
  x = churn_x, 
  y = churn_y,
  metric = "ROC",
  method = "ranger",
  trControl = myControl
)


# comparing models
# primary reason that train/test indices need to match when comparing two models:
# Because otherwise you wouldn't be doing a fair comparison of your models and your results could be due to chance.

# Create model_list
model_list <- list(item1 = model_glmnet, item2 = model_rf)

# Pass model_list to resamples(): resamples
resamples <- resamples(model_list)

# Summarize the results
summary(resamples)
# The resamples function gives us a bunch of options for comparing models


# the box-and-whisker plot allows you to compare the distribution of predictive accuracy (in this case AUC) for the two models
# In general, you want the model with the higher median AUC, as well as a smaller range between min and max AUC.

# Create bwplot
bwplot(resamples, metric = "ROC")


# Another useful plot for comparing models is the scatterplot, also known as the xy-plot. This plot shows you how similar the two models' performances are on different folds.
# It's particularly useful for identifying if one model is consistently better than the other across all folds, or if there are situations when the inferior model produces better predictions on a particular subset of the data.

# Create xyplot
xyplot(resamples, metric = "ROC")