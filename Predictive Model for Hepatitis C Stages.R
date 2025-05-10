# Load necessary libraries
library(utils)
library(psych)
library(caret)
library(tidyverse)
library(skimr)
library(stringr)
library(themis) 
library(vip) 
library(probably) 
library("ggplot2")            
library("GGally")  
library(corrplot)
library(randomForest)
library(pROC)
library(tidymodels)
library(ranger)
library(xgboost)

# 1. Data Acquisition
# URL of the zip file
url <- "https://archive.ics.uci.edu/static/public/571/hcv+data.zip"

# Download the zip file
download.file(url, destfile = "hcv_data.zip")

# Unzip the file
unzip("hcv_data.zip")

# Load data
data <- read.csv("hcvdat0.csv")
head(data)
str(data)

# Summary(data)
dim(data)

# Encoding catogeriocal variable
data <- data %>% 
  mutate(Diagnosis = if_else(str_detect(Category, "Donor"), "Donor", "Hepatitis")) %>%
  mutate(Diagnosis = factor(Diagnosis, levels = c("Hepatitis", "Donor"))) %>%
  relocate(Diagnosis, .before = Category) %>%
  select(-Category)
data <- subset(data, select = -c(X) )
head(data)


# 2. Data Exploration
# Bar plot 
ggplot(data=data,                             
       aes(x=Diagnosis, 
           fill=Diagnosis)) +                       
  geom_bar() +                                 
  scale_x_discrete(name = 'Diagnosis',       
                   labels=labs) + 
  scale_fill_discrete(name = 'Diagnosis') + 
  labs(x='Diagnosis',                       
       y='Count') + 
  theme_bw() 

# Relationship between Diagnosis and other Variables
data %>%
  gather(-Diagnosis, key = "var", value = "value") %>%
  ggplot(aes(x = value, y = Diagnosis, color = Diagnosis)) +
  geom_jitter(alpha = 0.5, size = 0.6) +
  stat_smooth() +
  facet_wrap(~ var, scales = "free") +
  theme_bw()

# Detection of outliers for continuous features
# Using BoxPlot to detect the presence of outliers 
boxplot(data[,c('Age','ALB','ALP','ALT','AST','BIL','CHE','CHOL','CREA','GGT', 'PROT')])

outlier_counts <- sapply(data[, c('Age','ALB','ALP','ALT','AST','BIL','CHE','CHOL','CREA','GGT', 'PROT')], function(x) {
  length(boxplot.stats(x)$out)
})

outlier_counts

# Correlation analysis
# Selecting only the numeric columns for correlation analysis
numeric_data <- data[, sapply(data, is.numeric)]

# Calculating the correlation matrix
correlation_matrix <- cor(numeric_data, use = "complete.obs") 

# Displaying the correlation matrix
print(correlation_matrix)

# Plotting the correlation matrix
corrplot(correlation_matrix, method = "color")

# Evaluation of distribution
# Selecting data columns
selected_columns <- c('Age','ALB','ALP','ALT','AST','BIL','CHE','CHOL','CREA','GGT', 'PROT')

# Creating subset of the data
data_subset <- select(data, all_of(selected_columns))

# Generating statistical summary
statistical_summary <- summary(data_subset)
print(statistical_summary)

# Function to create histogram and density plot
create_plots <- function(data, col) {
  p <- ggplot(data, aes_string(x = col)) +
    geom_histogram(aes(y = ..density..), binwidth = 1, fill = "salmon", alpha = 0.7) +
    geom_density(colour = "red", size = 1) +
    ggtitle(paste("Histogram and Density of", col)) +
    theme_minimal()
  print(p)
}

# Reshape the data for box plot 
long_data <- pivot_longer(data_subset, cols = everything(), names_to = "Variable", values_to = "Value")

# Apply the function to each selected column
lapply(names(data_subset), function(col) create_plots(data_subset, col))

# Box Plots for each selected variable
melted_data <- reshape2::melt(data_subset)
p <- ggplot(melted_data, aes(x = variable, y = value)) +
  geom_boxplot(fill = "lightblue", colour = "darkblue") +
  ggtitle("Box Plots of Selected Variables") +
  theme_minimal() +
  xlab("Variables") + ylab("Values")
print(p)

# 3. Data Cleaning & Shaping
# Identification of missing values
na_values <- sapply(data, function(x) sum(is.na(x)))

# Printing the number of missing values per column
print(na_values)

# Normalization feature values
# Normalization 
normalize <- function(x) {
  return((x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))
}

data_norm <- as.data.frame(lapply(imputed_data, function(x) if(is.numeric(x)) normalize(x) else x))

# 4. Model Construction
# Split the data into training and testing subsets 70:30 
set.seed(123) 
trainIndex <- createDataPartition(data_norm$Diagnosis, p = 0.7, list = FALSE)
df_train <- data_norm[trainIndex, ]
df_val <- data_norm[-trainIndex, ]

# Dummy Encoding for Nominal Predictors
m <- recipe(Diagnosis ~ ., data = df_train)

# Dummy Encoding for Nominal Predictors
dummy <- step_dummy(m, all_nominal_predictors())

# Normalization of Predictors
nor <- step_normalize(dummy, all_predictors())

# SMOTE for Balancing the Classes
s <- step_smote(nor, Diagnosis, over_ratio = 1, seed = 100)

# Creation of model A with proper data encoding: Logistic Regression
# Specification 
lr_model <- logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet")
# Specification 
lr_model <- logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet")
# Joining Model and Processing Recipe
lr <- workflow() %>% 
  add_model(lr_model) %>% 
  add_recipe(s)
lr_grid <- tibble(penalty = 10**seq(-4, 0, length.out = 30))

# Stratified, Repeated 10-fold Cross-Validation
cv_lr <- vfold_cv(df_train, strata = "Diagnosis", v = 10, repeats = 5)

# Metrics 
cls <- metric_set(roc_auc, j_index)

# Model Tuning
lr_res <- tune_grid(lr,
                    grid = lr_grid,
                    resamples = cv_lr,
                    control = control_grid(save_pred = TRUE),
                    metrics = cls)

# Best models ranked by AUC
best_mean_AUC <- show_best(lr_res, metric = "roc_auc")
best_mean_AUC

# Best models ranked by J-index
best_j_index <- show_best( lr_res, metric = "j_index")
best_j_index

# Best hyper-parameters
lr_best <- select_best(lr_res, metric = "roc_auc")
lr_best

# ROC-AUC of the best model
lr_auc <- lr_res %>% 
  collect_predictions(parameters = lr_best) %>% 
  roc_curve(Diagnosis, .pred_Hepatitis) %>% 
  mutate(model = "Logistic Regression")

autoplot(lr_auc)

# Final Model Fitting for Logistic Regression
final_lr_model <- finalize_model(lr_model, lr_best)
final_lr_wf <- workflow() %>% 
  add_model(final_lr_model) %>% 
  add_recipe(s)

# Fit the model on the training data
fitted_lr_model <- fit(final_lr_wf, data = df_train)

# Creation of model B with proper data encoding: Random Fores
# Specification 
rf_model <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 50) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")
# Joining Model and Processing Recipe
rf <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(s)
# Stratified, Repeated 10-fold Cross-Validation
cv <- vfold_cv(df_train, strata = "Diagnosis", v = 10, repeats = 5)
# Metrics 
cls <- metric_set(roc_auc, j_index)

# Model Tuning
rf_res <- tune_grid(rf,
                    grid = 25,
                    resamples = cv,
                    control = control_grid(save_pred = TRUE),
                    metrics = cls)

# Best models ranked by AUC
best_mean_AUC <- show_best(rf_res, metric = "roc_auc")
best_mean_AUC

# Best models ranked by J-index
best_j_index <- show_best( rf_res, metric = "j_index")
best_j_index

# Best hyper-parameters
rf_best <- select_best(rf_res, metric = "roc_auc")
rf_best

# ROC-AUC of the best model
rf_auc <- rf_res %>% 
  collect_predictions(parameters = rf_best) %>% 
  roc_curve(Diagnosis, .pred_Hepatitis) %>% 
  mutate(model = "Random Forest")

autoplot(rf_auc)

# Final Model Fitting for Random Forest
final_rf_model <- finalize_model(rf_model, rf_best)
final_rf_wf <- workflow() %>% 
  add_model(final_rf_model) %>% 
  add_recipe(s)

# Fit the model on the training data
fitted_rf_model <- fit(final_rf_wf, data = df_train)


# Creation of model C with proper data encoding: Boosted Trees model
set.seed(123)
# Model Specification
bt_model <- boost_tree(
  mtry = tune(),
  trees = 50,
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  stop_iter = tune()) %>%
  set_engine("xgboost") %>% 
  set_mode("classification")

# Joining Model and Processing Recipe
bt <- workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(s)
# Stratified, Repeated 10-fold Cross-Validation
cv_bt <- vfold_cv(df_train, strata = "Diagnosis", v = 10, repeats = 5)
# Metrics
cls <- metric_set(roc_auc, j_index)

# Model Tuning
bt_res <- tune_grid(bt,
                    grid = 25,
                    resamples = cv_bt,
                    control = control_grid(save_pred = TRUE),
                    metrics = cls)

# Best models ranked by AUC
best_mean_AUC <- show_best(bt_res, metric = "roc_auc")
best_mean_AUC

# Best models ranked by J-index
best_j_index <- show_best( bt_res, metric = "j_index")
best_j_index

# Best hyper-parameters
bt_best <- select_best(bt_res, metric = "roc_auc")
bt_best
# ROC-AUC of the best model
bt_auc <- bt_res %>% 
  collect_predictions(parameters = bt_best) %>% 
  roc_curve(Diagnosis, .pred_Hepatitis) %>% 
  mutate(model = "Boosted Trees")

autoplot(bt_auc)

# Final Model Fitting for Boosted Trees
final_bt_model <- finalize_model(bt_model, bt_best)
final_bt_wf <- workflow() %>% 
  add_model(final_bt_model) %>% 
  add_recipe(s)

# Fit the model on the training data
fitted_bt_model <- fit(final_bt_wf, data = df_train)

# 5. Model Evaluation
# Comparison of models and interpretation
bind_rows(rf_auc, lr_auc, bt_auc) %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) + 
  geom_path(lwd = 1, alpha = 0.8) +
  geom_abline(lty = 3) + 
  coord_equal() + 
  scale_color_viridis_d(alpha = 1,
                        begin = 0,
                        end = 1,
                        direction = 1,
                        option = "D",
                        aesthetics = "colour")

# 6. Model Tuning & Performance Improvement
ensemble_model_function <- function(lr_mod, rf_mod, bt_mod, test_data) {
  # Predictions from individual models
  predictions_lr <- predict(lr_mod, df_val, type = "prob")[, 1]
  predictions_rf <- predict(rf_mod, df_val, type = "prob")[, 1]
  predictions_bt <- predict(bt_mod, df_val, type = "prob")[, 1]
  
  # Convert probabilities to binary votes
  binary_vote_lr <- ifelse(predictions_lr >= 0.5, 1, 0)
  binary_vote_rf <- ifelse(predictions_rf >= 0.5, 1, 0)
  binary_vote_bt <- ifelse(predictions_bt >= 0.5, 1, 0)
  
  # Sum of binary votes
  combined_votes <- binary_vote_lr + binary_vote_rf + binary_vote_bt
  majority_vote <- ifelse(combined_votes >= 2, 1, 0)
  
  combined_predictions <- predictions_lr + predictions_rf + predictions_bt
  majority_vote <- ifelse(combined_predictions >= 2, 1, 0)
  #print(majority_vote)
  # Create a data frame to show individual model votes
  votes_df <- data.frame(lr_vote = binary_vote_lr, 
                         rf_vote = binary_vote_rf, 
                         bt_vote = binary_vote_bt, 
                         majority_vote = majority_vote)
  # Convert to factor with appropriate levels
  level_mapping <- c("Donor", "Hepatitis") 
  ensemble_factor <- factor(majority_vote, levels = c(0, 1), labels = level_mapping)
  
  # Confusion matrix
  c <- confusionMatrix(ensemble_factor, df_val$Diagnosis)
  print(c)
  
  return(list(ensemble_factor, votes_df))
}
cat("Ensemble model: ")
final_predictions <- ensemble_model_function(fitted_lr_model, fitted_rf_model, fitted_bt_model, df_val)
summary(final_predictions[[2]])

# Comparison of ensemble to individual models
evaluate_model <- function(model, test_data, test_labels) {
  # Predict probabilities
  predictions_prob <- predict(model, test_data, type = "prob")[, 1]
  binary_vote <- ifelse(predictions_prob >= 0.5, 1, 0)
  
  # Convert probabilities to binary predictions based on a threshold (e.g., 0.5)
  threshold <- 0.5
  predictions_binary <- ifelse(predictions_prob > threshold, 1, 0)
  
  # Converting to factor
  level_mapping <- c("Donor", "Hepatitis") 
  predictions_binary_factor <- factor(predictions_binary, levels = c(0, 1), labels = level_mapping)
  
  
  # Confusion matrix
  cm <- confusionMatrix(as.factor(predictions_binary_factor), as.factor(test_labels))
  print(cm)
  
  # Calculate metrics
  accuracy <- cm$overall['Accuracy']
  precision <- cm$byClass['Precision']
  recall <- cm$byClass['Recall']
  f1 <- 2 * (precision * recall) / (precision + recall)
  
  # Return a list of metrics
  return(list(accuracy = accuracy, precision = precision, recall = recall, f1 = f1))
}

cat("Logistic Regression: ")
results_lr <- evaluate_model(fitted_lr_model, df_val, df_val$Diagnosis)
cat("Random Forest: ")
results_rf <- evaluate_model(fitted_rf_model, df_val, df_val$Diagnosis)
cat("Boosted Tree: ")
results_bt <- evaluate_model(fitted_bt_model, df_val, df_val$Diagnosis)
