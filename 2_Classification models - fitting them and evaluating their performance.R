# classification models are used to predict a categorical (qualitative) variable

library(mlbench)  # contains the Sonar dataset
library(caTools)  # for computing ROC curves

# will be using a 60% training set and a 40% test set

# Get the number of observations
n_obs <- nrow(Sonar)

# Shuffle row indices: permuted_rows
permuted_rows <- sample(n_obs)

# Randomly order data: Sonar
Sonar_shuffled <- Sonar[permuted_rows, ]

# Identify row to split on: split
split <- round(n_obs * 0.6)

# Create train
train <- Sonar_shuffled[1:split, ]

# Create test
test <- Sonar_shuffled[(split + 1):n_obs, ]

# Fit glm model: model
model <- glm(Class ~ ., family = "binomial", train)
# Don't worry about warnings; these are common on smaller datasets and usually don't cause any issues. 
# They typically mean your dataset is perfectly separable, which can cause problems for the math behind the model, but R's glm() function is almost always robust enough to handle this case with no problems.

# Predict on test: p
p <- predict(model, test, type = "response")


# confusion matrix: a useful tool for evaluating binary classification models
# a confusion matrix is a very useful tool for calibrating the output of a model and examining all possible outcomes of your predictions (true positive, true negative, false positive, false negative)

# If p exceeds threshold of 0.5, M else R: m_or_r
m_or_r <- ifelse(p > 0.5, "M", "R")

# Convert to factor: p_class
p_class <- factor(m_or_r, levels = levels(test[["Class"]]))

# Create confusion matrix
confusionMatrix(p_class, test[["Class"]])

# If p exceeds threshold of 0.9, M else R: m_or_r
m_or_r <- ifelse(p > 0.9, "M", "R")

# Convert to factor: p_class
p_class <- factor(m_or_r, levels = levels(test[["Class"]]))

# Create confusion matrix
confusionMatrix(p_class, test[["Class"]])

# If p exceeds threshold of 0.1, M else R: m_or_r
m_or_r <- ifelse(p > 0.1, "M", "R")

# Convert to factor: p_class
p_class <- factor(m_or_r, levels = levels(test[["Class"]]))

# Create confusion matrix
confusionMatrix(p_class, test[["Class"]])

# a more systematic approach to evaluating classification thresholds:
# first, plot true/false positives rate at every possible threshold
# then visualize tradeoffs between two extremes (100% true positive vs 0% false positive)
# result is a ROC curve (receiver operating characteristic; outdated WW2 term)

# A ROC curve is a really useful shortcut for summarizing the performance of a classifier over all possible thresholds. 
# This saves you a lot of tedious work computing class predictions for many different thresholds and examining the confusion matrix for each.

# colAUC() for calculating ROC curve; returns a score called AUC (area under the curve) and also produces a visual plot

# Predict on test: p
p <- predict(model, test, type = "response")

# Make ROC curve
colAUC(p, test[["Class"]], plotROC = TRUE)


# AUC (area under the curve): single-number summary of model accuracy 
# AUC range is 0 to 1: exactly 1 for a perfect model; average AUC for a random guessing model is 0.5
# most models fall between 0.5 and 1.0

# Create trainControl object: myControl
myControl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT! (needed to calculate AUC)
  verboseIter = TRUE
)

# Train glm with custom trainControl: model
model <- train(
  Class ~ ., 
  Sonar, 
  method = "glm",
  trControl = myControl
)
# Note that fitting a glm with caret often produces warnings about convergence or probabilities. 
# These warnings can almost always be safely ignored, as you can use the glm's predictions to validate whether the model is accurate enough for your task. 

# Print model to console
model