# The caret package automates supervised learning a.k.a. predictive modeling
# Two types of predictive models:
# 1. Classification -> Qualitative (e.g. species of a flower)
# 2. Regression -> Quantitative (e.g. price of a diamond)

# RMSE is a metric for regression models to evaluate how well the model works
# RMSE is a measure of the model's average error
# calculating in-sample RMSE is generally too optimistic and leads to overfitting
# a better approach is to use an out-of-sample estimate (caret does this) for an estimate of how well the model performs on new data

library(caret)
library(ggplot2)  # contains the diamonds dataset

# in-sample example:

# Fit lm model: model
model <- lm(price ~ ., diamonds)
# used to predict the price based on all the other columns of data in the diamonds dataset

# Predict on full data: p
p <- predict(model, diamonds)

# Compute errors: error
error <- p - diamonds[["price"]]
# errors = predicted - actual

# Calculate RMSE
sqrt(mean(error ^ 2))  # 1129.843
# means the model is off by $1129.84 on average

# out-of-sample example:

# One way you can take a train/test split of a dataset is to order the dataset randomly, then divide it into the two sets. 
# This ensures that the training set and test set are both random samples and that any biases in the ordering of the dataset (e.g. if it had originally been ordered by price or size) are not retained in the samples.
# First, you set a random seed so that your work is reproducible and you get the same random split each time you run your script: set.seed(42)
# Next, you use the sample() function to shuffle the row indices of the diamonds dataset. You can later use these indices to reorder the dataset. rows <- sample(nrow(diamonds))
# Finally, you can use this random vector to reorder the diamonds dataset: diamonds <- diamonds[rows, ]

# Set seed
set.seed(42)

# Shuffle row indices: rows
rows <- sample(nrow(diamonds))

# Randomly order data
shuffled_diamonds <- diamonds[rows, ]

# Now that your dataset is randomly ordered, you can split the first 80% of it into a training set, and the last 20% into a test set. split <- round(nrow(mydata) * 0.80)
# You can then use this point to break off the first 80% of the dataset as a training set: mydata[1:split, ]
# And then you can use that same point to determine the test set: mydata[(split + 1):nrow(mydata), ]

# Determine row to split on: split
split <- round(nrow(diamonds) * 0.80)

# Create train
train <- diamonds[1:split, ]

# Create test
test <- diamonds[(split + 1):nrow(diamonds), ]

# Fit lm model on train: model
model <- lm(price ~ ., train)

# Predict on test: p
p <- predict(model, test)

# Compute errors: error
error <- p - test[["price"]]

# Calculate RMSE
sqrt(mean(error^2))  # 796.8922


# use multiple test sets and average the out-of-sample error (gives a better estimate of the true out-of-sample error)
# cross-validation does this by creating 10 test sets (called "folds") from our dataset 
# cross-validation is only used to estimate the out-of-sample error for the model; once you know this, you refit your model on the full training dataset
# train() function helps with cross-validation

# Fit lm model using 10-fold CV: model
model <- train(
  price ~ ., 
  diamonds,
  method = "lm",
  trControl = trainControl(
    method = "cv", 
    number = 10,
    verboseIter = TRUE
  )
)
# Caret does all the work of splitting test sets and calculating RMSE for you

# Print model to console
model


# THE BOSTON DATASET ISN'T LOADED HERE SO THE CODE WON'T WORK

# Fit lm model using 5-fold CV: model
model <- train(
  medv ~ ., 
  Boston,
  method = "lm",
  trControl = trainControl(
    method = "cv", 
    number = 5,
    verboseIter = TRUE
  )
)

# Print model to console
model


# Repeated cross-validation gives you a better estimate of the test-set error. 

# Fit lm model using 5 x 5-fold CV: model
model <- train(
  medv ~ ., 
  Boston,
  method = "lm",
  trControl = trainControl(
    method = "repeatedcv", 
    number = 5,
    repeats = 5, 
    verboseIter = TRUE
  )
)

# Print model to console
model

# Predict on full Boston dataset
predict(model, Boston)