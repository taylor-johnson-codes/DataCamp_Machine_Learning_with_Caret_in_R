# random forests are a popular type of machine learning model
# Random forest models are much more flexible than linear models, and can model complicated nonlinear effects as well as automatically capture interactions between variables.
# They tend to give very good results on real world data.

library(caret)
library(ranger)  # a rewrite of R's classic randomForest package 

# THE WINE & OVERFIT DATASETS AREN'T LOADED HERE SO THE CODE WON'T WORK

# Fit random forest: model
model <- train(
  quality ~ .,
  tuneLength = 1,
  data = wine, 
  method = "ranger",
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    verboseIter = TRUE
  )
)

# Print model to console
model


# random forests require tuning
# Random forest models have a primary tuning parameter of mtry, which controls how many variables are exposed to the splitting search routine at each split.
# For example, suppose that a tree has a total of 10 splits and mtry = 2. This means that there are 10 samples of 2 predictors each time a split is evaluated.

# Fit random forest: model
model <- train(
  quality ~ .,
  tuneLength = 3,
  data = wine, 
  method = "ranger",
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    verboseIter = TRUE
  )
)
# You can adjust the tuneLength variable to make a trade-off between runtime and how deep you want to grid-search the model.

# Print model to console
model

# Plot model
plot(model)


# Now that you've explored the default tuning grids provided by the train() function, let's customize your models a bit more.
# You can provide any number of values for mtry, from 2 up to the number of columns in the dataset. 
# In practice, there are diminishing returns for much larger values of mtry, so you will use a custom tuning grid that explores 2 simple models (mtry = 2 and mtry = 3) as well as one more complicated model (mtry = 7).

# Define the tuning grid: tuneGrid
tuneGrid <- data.frame(
  .mtry = c(2, 3, 7),
  .splitrule = "variance",
  .min.node.size = 5
)

# Fit random forest: model
model <- train(
  quality ~ .,
  tuneGrid = tuneGrid,
  data = wine, 
  method = "ranger",
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    verboseIter = TRUE
  )
)

# Print model to console
model

# Plot model
plot(model)


# glmnet: extension of glm models with built-in variable selection; attempts to find a simple model; pairs well with random forest models
# two primary forms:
# 1. lasso regression: penalizes number of non-zero coefficients
# 2. ridge regression: penalizes absolute magnitude of coefficients

# glmnet is an extension of the generalized linear regression model (or glm) that places constraints on the magnitude of the coefficients to prevent overfitting.
# This is more commonly known as "penalized" regression modeling and is a very useful technique on datasets with many predictors and few values.
# glmnet is capable of fitting two different kinds of penalized models, controlled by the alpha parameter:
# Ridge regression (or alpha = 0)
# Lasso regression (or alpha = 1)

# Classification problems are a little more complicated than regression problems because you have to provide a custom summaryFunction to the train() function to use the AUC metric to rank your models.

# Create custom trainControl: myControl
myControl <- trainControl(
  method = "cv", 
  number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT! (needed to calculate AUC)
  verboseIter = TRUE
)

# Now that you have a custom trainControl object, fit a glmnet model to the "don't overfit" dataset.

# Fit glmnet model: model
model <- train(
  y ~ ., 
  overfit,
  method = "glmnet",
  trControl = myControl
)

# Print model to console
model

# Print maximum ROC statistic
max(model[["results"]][["ROC"]])


# The glmnet model fits many models at once. You can exploit this by passing a large number of lambda values, which control the amount of penalization in the model.
# train() is smart enough to only fit one model per alpha value and pass all of the lambda values at once for simultaneous fitting.

# Train glmnet with custom trainControl and tuning: model
model <- train(
  y ~ ., 
  overfit,
  tuneGrid = expand.grid(
    alpha = 0:1,
    lambda = seq(0.0001, 1, length = 20)
  ),
  method = "glmnet",
  trControl = myControl
)

# Print model to console
model

# Print maximum ROC statistic
max(model[["results"]][["ROC"]])