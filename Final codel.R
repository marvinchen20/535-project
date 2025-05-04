# 1) Load required libraries and set seed
library(dplyr)    # for data wrangling
library(caret)    # for preprocessing functions
library(dplyr)
library(tibble)
library(glmnet)
set.seed(41)

################## data prepared #######################

# 2) Read in the raw Titanic training data
train_path <- "D:/BU PHD/535/titanic/train.csv"    # adjust to your filepath
df <- read.csv(train_path)

# 3) Select only the columns we’ll use
df <- df %>%
  select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare)

# 4) Convert key columns to factors
df <- df %>%
  mutate(
    Survived = factor(Survived, levels = c(0,1), labels = c("No","Yes")),
    Pclass   = factor(Pclass),
    Sex      = factor(Sex)
  )

# 5) Impute missing numerics (Age) and then center & scale all numerics
num_preds <- c("Age","SibSp","Parch","Fare")
preproc   <- preProcess(
  df[, num_preds],
  method = c("medianImpute", "center", "scale")
)
df[, num_preds] <- predict(preproc, df[, num_preds])

# 6) Inspect the cleaned data
head(df)

# Option B: programmatically into a named list
train_by_pclass <- split(df, df$Pclass)

# can then access by:
#   train_by_pclass[["1"]]
#   train_by_pclass[["2"]]
#   train_by_pclass[["3"]]

# And inspect:
sapply(train_by_pclass, nrow)

##########################   Variable selection ######################
# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: VARIABLE SELECTION (Stepwise, Forward & Backward) BY Pclass
# ─────────────────────────────────────────────────────────────────────────────

library(dplyr)
library(tibble)

selection_results <- lapply(names(train_by_pclass), function(pc) {
  dat <- train_by_pclass[[pc]]
  
  # build null & full models
  null_mod <- glm(Survived ~ 1,
                  data   = dat,
                  family = binomial)
  full_mod <- glm(Survived ~ Sex + Age + SibSp + Parch + Fare,
                  data   = dat,
                  family = binomial)
  
  # a) stepwise (both directions)
  step_both    <- step(null_mod,
                       scope     = list(lower = null_mod, upper = full_mod),
                       direction = "both",
                       trace     = FALSE)
  vars_both    <- attr(terms(step_both), "term.labels")
  
  # b) forward
  step_forward <- step(null_mod,
                       scope     = list(lower = null_mod, upper = full_mod),
                       direction = "forward",
                       trace     = FALSE)
  vars_forward <- attr(terms(step_forward), "term.labels")
  
  # c) backward
  step_backward<- step(full_mod,
                       direction = "backward",
                       trace     = FALSE)
  vars_backward<- attr(terms(step_backward), "term.labels")
  
  tibble(
    Pclass        = pc,
    method        = c("step_both", "step_forward", "step_backward"),
    variables     = c(
      paste(vars_both,     collapse = ", "),
      paste(vars_forward,  collapse = ", "),
      paste(vars_backward, collapse = ", ")
    )
  )
})

# combine into one table
sel_tbl <- bind_rows(selection_results)

# if you’d like to merge identical variable‐sets into one row per Pclass:
merged_tbl <- sel_tbl %>%
  group_by(Pclass, variables) %>%
  summarize(
    methods = paste(method, collapse = ", "),
    .groups = "drop"
  )

# show results
sel_tbl



####lasso#####
dat1 <- train_by_pclass[["1"]]
dat2 <- train_by_pclass[["2"]]
dat3 <- train_by_pclass[["3"]]
# 1) Subset data for Pclass 1
# 2) Create model matrix and response
x1 <- model.matrix(Survived ~ Sex + Age + SibSp + Parch + Fare, data = dat1)[, -1]
y1 <- as.numeric(dat1$Survived == "Yes")


# Define a sequence of lambda values to try
lambda_seq <- c(1, 0.1, 0.05, 0.01, 0.005, 0.001)

# Loop over lambda values
for (lam in lambda_seq) {
  fit <- glmnet(x1, y1, family = "binomial", alpha = 1, lambda = lam)
  coefs <- coef(fit)
  vars <- rownames(coefs)[which(coefs != 0)]
  vars <- setdiff(vars, "(Intercept)")
  
  cat(sprintf("LASSO selected (lambda = %.4f): %s\n", lam,
              if (length(vars)) paste(vars, collapse = ", ") else "(none)"))
}

#### LASSO for Pclass 2 ####
x2 <- model.matrix(Survived ~ Sex + Age + SibSp + Parch + Fare, data = dat2)[, -1]
y2 <- as.numeric(dat2$Survived == "Yes")

cat("\n=== LASSO for Pclass 2 ===\n")
for (lam in lambda_seq) {
  fit <- glmnet(x2, y2, family = "binomial", alpha = 1, lambda = lam)
  coefs <- coef(fit)
  vars <- rownames(coefs)[which(coefs != 0)]
  vars <- setdiff(vars, "(Intercept)")
  cat(sprintf("LASSO selected (lambda = %.4f): %s\n", lam,
              if (length(vars)) paste(vars, collapse = ", ") else "(none)"))
}


#### LASSO for Pclass 3 ####
x3 <- model.matrix(Survived ~ Sex + Age + SibSp + Parch + Fare, data = dat3)[, -1]
y3 <- as.numeric(dat3$Survived == "Yes")

cat("\n=== LASSO for Pclass 3 ===\n")
for (lam in lambda_seq) {
  fit <- glmnet(x3, y3, family = "binomial", alpha = 1, lambda = lam)
  coefs <- coef(fit)
  vars <- rownames(coefs)[which(coefs != 0)]
  vars <- setdiff(vars, "(Intercept)")
  cat(sprintf("LASSO selected (lambda = %.4f): %s\n", lam,
              if (length(vars)) paste(vars, collapse = ", ") else "(none)"))
}

merged_tbl


#########

selection_results <- lapply(names(train_by_pclass), function(pc) {
  dat <- train_by_pclass[[pc]]
  
  # Build null & full models
  null_mod <- glm(Survived ~ 1, data = dat, family = binomial)
  full_mod <- glm(Survived ~ Sex + Age + SibSp + Parch + Fare, data = dat, family = binomial)
  
  # Stepwise both
  step_both <- step(null_mod, scope = list(lower = null_mod, upper = full_mod),
                    direction = "both", trace = FALSE)
  vars_both <- attr(terms(step_both), "term.labels")
  
  # Forward
  step_forward <- step(null_mod, scope = list(lower = null_mod, upper = full_mod),
                       direction = "forward", trace = FALSE)
  vars_forward <- attr(terms(step_forward), "term.labels")
  
  # Backward
  step_backward <- step(full_mod, direction = "backward", trace = FALSE)
  vars_backward <- attr(terms(step_backward), "term.labels")
  
  # LASSO (no CV, fixed lambda)
  x <- model.matrix(Survived ~ Sex + Age + SibSp + Parch + Fare, data = dat)[, -1]
  y <- as.numeric(dat$Survived == "Yes")
  lasso_fit <- glmnet(x, y, family = "binomial", alpha = 1, lambda = 0.05)
  coef_lasso <- coef(lasso_fit)
  vars_lasso <- rownames(coef_lasso)[which(coef_lasso != 0)]
  vars_lasso <- setdiff(vars_lasso, "(Intercept)")
  
  # Output as tibble
  tibble::tibble(
    Pclass = pc,
    method = c("step_both", "step_forward", "step_backward", "lasso"),
    variables = c(
      paste(vars_both, collapse = ", "),
      paste(vars_forward, collapse = ", "),
      paste(vars_backward, collapse = ", "),
      paste(vars_lasso, collapse = ", ")
    )
  )
})

# Combine into one table

selection_table <- dplyr::bind_rows(selection_results)
# Replace any dummy-coded sex variable with just "Sex"
# Apply cleanup for "Sexmale" → "Sex" per entry
selection_table <- selection_table %>%
  mutate(
    variables = variables %>%
      strsplit(split = ",\\s*") %>%              # Split string by comma + space
      lapply(function(v) unique(gsub("^Sex.*$", "Sex", v))) %>%  # Replace any 'Sex*' with 'Sex'
      sapply(function(v) paste(v, collapse = ", "))              # Collapse back to string
  )



library(dplyr)

merged_selection_table <- selection_table %>%
  group_by(Pclass, variables) %>%
  summarise(
    methods = paste(sort(method), collapse = ", "),
    .groups = "drop"
  )

print(merged_selection_table, n = nrow(merged_selection_table))

################# model fit ####################

###### SVM ######
library(caret)
library(e1071)
library(dplyr)

# set up control
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# initialize list to store results
svm_models <- list()

for (i in seq_len(nrow(merged_selection_table))) {
  pc    <- merged_selection_table$Pclass[i]
  vars  <- unlist(strsplit(merged_selection_table$variables[i], ",\\s*"))
  dat   <- train_by_pclass[[pc]]
  
  # build formula like Survived ~ Sex + Age + ...
  fmla <- reformulate(vars, response = "Survived")
  
  # train SVM
  model <- train(fmla, data = dat, method = "svmLinear", metric = "ROC", trControl = ctrl)
  
  # store by Pclass and variable set
  svm_models[[paste0("Pclass", pc, "_", gsub(", ", "_", merged_selection_table$variables[i]))]] <- model
}

for (name in names(svm_models)) {
  cat("\n=== ", name, " ===\n")
  print(svm_models[[name]])
}

######### RF ##########

library(randomForest)
library(caret)

# control settings (same as SVM)
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# initialize list to store RF models
rf_models <- list()

for (i in seq_len(nrow(merged_selection_table))) {
  pc    <- merged_selection_table$Pclass[i]
  vars  <- unlist(strsplit(merged_selection_table$variables[i], ",\\s*"))
  dat   <- train_by_pclass[[pc]]
  
  # build formula like Survived ~ Sex + Age + ...
  fmla <- reformulate(vars, response = "Survived")
  
  # train Random Forest
  model <- train(fmla, data = dat, method = "rf", metric = "ROC", trControl = ctrl, tuneLength = 5)
  
  # store model with identifier
  rf_models[[paste0("Pclass", pc, "_", gsub(", ", "_", merged_selection_table$variables[i]))]] <- model
}

# Print summary of each RF model
for (name in names(rf_models)) {
  cat("\n=== ", name, " ===\n")
  print(rf_models[[name]])
}

# initialize list to store logistic regression models
glm_models <- list()

for (i in seq_len(nrow(merged_selection_table))) {
  pc    <- merged_selection_table$Pclass[i]
  vars  <- unlist(strsplit(merged_selection_table$variables[i], ",\\s*"))
  dat   <- train_by_pclass[[pc]]
  
  # build formula like: Survived ~ Sex + Age + ...
  fmla <- reformulate(vars, response = "Survived")
  
  # fit logistic regression
  model <- glm(fmla, data = dat, family = binomial)
  
  # store model with identifier
  glm_models[[paste0("Pclass", pc, "_", gsub(", ", "_", merged_selection_table$variables[i]))]] <- model
}
# initialize list to store logistic regression models
glm_models <- list()

for (i in seq_len(nrow(merged_selection_table))) {
  pc    <- merged_selection_table$Pclass[i]
  vars  <- unlist(strsplit(merged_selection_table$variables[i], ",\\s*"))
  dat   <- train_by_pclass[[pc]]
  
  # build formula like: Survived ~ Sex + Age + ...
  fmla <- reformulate(vars, response = "Survived")
  
  # fit logistic regression
  model <- glm(fmla, data = dat, family = binomial)
  
  # store model with identifier
  glm_models[[paste0("Pclass", pc, "_", gsub(", ", "_", merged_selection_table$variables[i]))]] <- model
}

# Select Pclass = 3 data and its model formula
dat_p3 <- df %>% filter(Pclass == "3")
glm_vars <- c("Sex", "Age", "SibSp", "Parch", "Fare")

# Remove "Parch"
glm_vars_no_parch <- setdiff(glm_vars, "Parch")

# Build formula without Parch
fmla_no_parch <- as.formula(paste("Survived ~", paste(glm_vars_no_parch, collapse = " + ")))

# Fit logistic model
glm_p3_no_parch <- glm(fmla_no_parch, data = dat_p3, family = binomial)

# Store in glm_models list
glm_models[["3_noParch"]] <- glm_p3_no_parch


for (name in names(glm_models)) {
  cat("\n=== ", name, " ===\n")
  print(summary(glm_models[[name]]))
}


library(pROC)
par(mfrow = c(2, 2))  # adjust layout as needed

for (name in names(glm_models)) {
  model <- glm_models[[name]]
  dat   <- model$data
  probs <- predict(model, type = "response")
  
  roc_obj <- roc(dat$Survived, probs, quiet = TRUE)
  
  plot(roc_obj, main = paste("ROC -", name))
  cat(sprintf("%s: AUC = %.3f\n", name, auc(roc_obj)))
}

library(caret)
library(pROC)
library(dplyr)

# Setup cross-validation
cv_ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Create a list of training sets
train_sets <- list(
  `1` = train_by_pclass[["1"]],
  `2` = train_by_pclass[["2"]],
  `3` = train_by_pclass[["3"]]
)

# Your final merged selection table (assuming it's already created and named `merged_selection_table`)
# Iterate over it and fit models
cv_results <- list()

for (i in seq_len(nrow(merged_selection_table))) {
  row <- merged_selection_table[i, ]
  pc <- row$Pclass
  vars <- unlist(strsplit(row$variables, ",\\s*"))
  fmla <- reformulate(vars, response = "Survived")
  dat <- train_sets[[pc]]
  
  cat("\n==========\nPclass", pc, "| Vars:", row$variables, "\n")
  
  # GLM
  glm_fit <- train(
    fmla, data = dat,
    method = "glm",
    family = "binomial",
    metric = "ROC",
    trControl = cv_ctrl
  )
  
  # SVM
  svm_fit <- train(
    fmla, data = dat,
    method = "svmLinear",
    metric = "ROC",
    trControl = cv_ctrl
  )
  
  # RF
  rf_fit <- train(
    fmla, data = dat,
    method = "rf",
    tuneLength = 5,
    metric = "ROC",
    trControl = cv_ctrl
  )
  
  # Save all
  cv_results[[paste0("Pclass", pc, "_", gsub(",\\s*", "_", row$variables))]] <- list(
    glm = glm_fit,
    svm = svm_fit,
    rf  = rf_fit
  )
}



########### Some additive model #########
library(caret)
library(e1071)
library(randomForest)

train_class2 <- train_by_pclass[["2"]]
train_class3 <- train_by_pclass[["3"]]

# CV control
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,savePredictions = "final" )

# ─── Pclass 2: Survived ~ Sex + Age ──────────────────────────────────────────────
formula_2 <- Survived ~ Sex + Age

special_models_class2 <- list(
  glm = train(formula_2, data = train_class2, method = "glm", family = "binomial", trControl = ctrl, metric = "ROC"),
  svm = train(formula_2, data = train_class2, method = "svmLinear", trControl = ctrl, metric = "ROC"),
  rf  = train(formula_2, data = train_class2, method = "rf", trControl = ctrl, metric = "ROC")
)

# ─── Pclass 3: Survived ~ Sex + Age + SibSp ──────────────────────────────────────
formula_3 <- Survived ~ Sex + Age + SibSp

special_models_class3 <- list(
  glm = train(formula_3, data = train_class3, method = "glm", family = "binomial", trControl = ctrl, metric = "ROC"),
  svm = train(formula_3, data = train_class3, method = "svmLinear", trControl = ctrl, metric = "ROC"),
  rf  = train(formula_3, data = train_class3, method = "rf", trControl = ctrl, metric = "ROC")
)

# Store them together if needed
special_models <- list(
  Pclass2_Sex_Age     = special_models_class2,
  Pclass3_Sex_Age_SibSp = special_models_class3
)

# Add to cv_results under clear new names
cv_results[["Pclass2_Sex_Age"]] <- special_models_class2
cv_results[["Pclass3_Sex_Age_SibSp"]] <- special_models_class3



############ ROC Test ############
library(ggplot2)
library(pROC)

# Helper function to extract ROC curve from a model
get_roc_df <- function(model, label) {
  roc_obj <- roc(model$pred$obs, model$pred$Yes)
  data.frame(
    FPR = 1 - roc_obj$specificities,
    TPR = roc_obj$sensitivities,
    model = label,
    AUC = round(auc(roc_obj), 3)
  )
}

# Collect ROC data from all models
roc_curves <- list()

for (name in names(cv_results)) {
  group <- cv_results[[name]]
  roc_curves[[paste0(name, "_GLM")]] <- get_roc_df(group$glm, paste0(name, "_GLM"))
  roc_curves[[paste0(name, "_SVM")]] <- get_roc_df(group$svm, paste0(name, "_SVM"))
  roc_curves[[paste0(name, "_RF")]]  <- get_roc_df(group$rf,  paste0(name, "_RF"))
}

# Combine into one data frame
roc_df <- do.call(rbind, roc_curves)

# Plot ROC
ggplot(roc_df, aes(x = FPR, y = TPR, color = model)) +
  geom_line(size = 1) +
  geom_abline(lty = 2) +
  theme_minimal() +
  labs(title = "ROC Curves for GLM / SVM / RF",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(ncol = 1)) +
  facet_wrap(~model, scales = "free")


### AUC table ###




### AUC  table all ###
# Function to extract variable names from a model object
extract_vars <- function(model) {
  if (inherits(model, "train") && !is.null(model$terms)) {
    vars <- attr(terms(model), "term.labels")
    return(paste(vars, collapse = ", "))
  } else {
    return(NA)
  }
}

# Build AUC results using actual model formulas
auc_details <- lapply(names(cv_results), function(name) {
  pclass <- gsub("Pclass", "", strsplit(name, "_")[[1]][1])
  models <- cv_results[[name]]
  
  rows <- list()
  
  if (!is.null(models$glm)) {
    auc <- tryCatch({
      round(pROC::auc(models$glm$pred$obs, models$glm$pred$Yes), 3)
    }, error = function(e) NA)
    vars <- extract_vars(models$glm)
    rows[[length(rows) + 1]] <- data.frame(Pclass = pclass, Variables = vars,
                                           Model = "GLM", AUC = auc)
  }
  
  if (!is.null(models$svm)) {
    auc <- tryCatch({
      round(pROC::auc(models$svm$pred$obs, models$svm$pred$Yes), 3)
    }, error = function(e) NA)
    vars <- extract_vars(models$svm)
    rows[[length(rows) + 1]] <- data.frame(Pclass = pclass, Variables = vars,
                                           Model = "SVM", AUC = auc)
  }
  
  if (!is.null(models$rf)) {
    auc <- tryCatch({
      round(pROC::auc(models$rf$pred$obs, models$rf$pred$Yes), 3)
    }, error = function(e) NA)
    vars <- extract_vars(models$rf)
    rows[[length(rows) + 1]] <- data.frame(Pclass = pclass, Variables = vars,
                                           Model = "RF", AUC = auc)
  }
  
  bind_rows(rows)
})

auc_long <- bind_rows(auc_details) %>%
  arrange(as.integer(Pclass), Variables, Model)

# Show updated result
print(auc_long)

### AUC Additive Model ###
# Add special entries manually
special_auc_rows <- list()

# Pclass 2: Sex + Age
if (!is.null(special_models_class2$glm)) {
  special_auc_rows[[length(special_auc_rows) + 1]] <- data.frame(
    Pclass   = "2",
    Variables = "Sex, Age",
    Model    = "GLM",
    AUC      = round(pROC::auc(special_models_class2$glm$pred$obs,
                               special_models_class2$glm$pred$Yes), 3)
  )
}
if (!is.null(special_models_class2$svm)) {
  special_auc_rows[[length(special_auc_rows) + 1]] <- data.frame(
    Pclass   = "2",
    Variables = "Sex, Age",
    Model    = "SVM",
    AUC      = round(pROC::auc(special_models_class2$svm$pred$obs,
                               special_models_class2$svm$pred$Yes), 3)
  )
}
if (!is.null(special_models_class2$rf)) {
  special_auc_rows[[length(special_auc_rows) + 1]] <- data.frame(
    Pclass   = "2",
    Variables = "Sex, Age",
    Model    = "RF",
    AUC      = round(pROC::auc(special_models_class2$rf$pred$obs,
                               special_models_class2$rf$pred$Yes), 3)
  )
}

# Pclass 3: Sex + Age + SibSp
if (!is.null(special_models_class3$glm)) {
  special_auc_rows[[length(special_auc_rows) + 1]] <- data.frame(
    Pclass   = "3",
    Variables = "Sex, Age, SibSp",
    Model    = "GLM",
    AUC      = round(pROC::auc(special_models_class3$glm$pred$obs,
                               special_models_class3$glm$pred$Yes), 3)
  )
}
if (!is.null(special_models_class3$svm)) {
  special_auc_rows[[length(special_auc_rows) + 1]] <- data.frame(
    Pclass   = "3",
    Variables = "Sex, Age, SibSp",
    Model    = "SVM",
    AUC      = round(pROC::auc(special_models_class3$svm$pred$obs,
                               special_models_class3$svm$pred$Yes), 3)
  )
}
if (!is.null(special_models_class3$rf)) {
  special_auc_rows[[length(special_auc_rows) + 1]] <- data.frame(
    Pclass   = "3",
    Variables = "Sex, Age, SibSp",
    Model    = "RF",
    AUC      = round(pROC::auc(special_models_class3$rf$pred$obs,
                               special_models_class3$rf$pred$Yes), 3)
  )
}

# Combine and add to existing auc_long
auc_long <- bind_rows(auc_long, bind_rows(special_auc_rows)) %>%
  arrange(as.integer(Pclass), Variables, Model)

# View full table
print(auc_long)







###### Final Model  #########

library(dplyr)

# Step 1: Find the best model per Pclass (highest AUC)
best_models <- auc_long %>%
  group_by(Pclass) %>%
  slice_max(order_by = AUC, n = 1, with_ties = FALSE) %>%
  ungroup()

print(best_models)





############ Test #################
# Load test data
test_raw <- read.csv("D:/BU PHD/535/titanic/test.csv")

# Preprocess: same columns, factor levels as training
test_df <- test_raw %>%
  select(PassengerId, Pclass, Sex, Age, SibSp, Parch, Fare) %>%
  mutate(
    Pclass = factor(Pclass, levels = levels(df$Pclass)),
    Sex    = factor(Sex, levels = levels(df$Sex))
  )

# Apply same preprocessing (imputation + scaling)
test_df[, c("Age", "SibSp", "Parch", "Fare")] <- predict(preproc, test_df[, c("Age", "SibSp", "Parch", "Fare")])


# ######## model 1########
# # Initialize result list
# test_preds <- list()
# 
# for (i in seq_len(nrow(best_models))) {
#   row     <- best_models[i, ]
#   pclass  <- row$Pclass
#   model   <- row$Model
#   var_str <- row$Variables
#   vars    <- trimws(strsplit(var_str, ",")[[1]])
#   
#   # Subset test set by Pclass
#   test_sub <- test_df %>% filter(Pclass == pclass)
#   
#   # Prepare formula
#   fmla <- as.formula(paste("Survived ~", paste(vars, collapse = " + ")))
#   
#   # Retrieve trained model from cv_results
#   model_key <- paste0("Pclass", pclass, "_", gsub(", ", "_", var_str))
#   fitted_model <- cv_results[[model_key]][[tolower(model)]]
#   
#   # Predict survival probabilities
#   if (model == "GLM") {
#     probs <- predict(fitted_model, newdata = test_sub, type = "prob")[, "Yes"]
#   } else if (model %in% c("SVM", "RF")) {
#     probs <- predict(fitted_model, newdata = test_sub, type = "prob")[, "Yes"]
#   }
#   
#   # Save predictions
#   test_preds[[pclass]] <- tibble(
#     PassengerId = test_sub$PassengerId,
#     Survived    = ifelse(probs > 0.5, 1L, 0L)
#   )
# }
# 
# 
# 
# # Combine all class predictions
# submission <- bind_rows(test_preds) %>%
#   arrange(PassengerId)
# 
# 
# 
# # Save as CSV
# write.csv(submission, "D:/BU PHD/535/titanic/final_submission_best_models.csv", row.names = FALSE)
# 
# print(head(submission))

############ model 2###################
# Step 1: Define formulas for each class
formulas <- list(
  "1" = Survived ~ Sex + Age,
  "2" = Survived ~ Sex + Age + Parch,
  "3" = Survived ~ Sex + Age + SibSp
)

# Step 2: Reference training sets
train_data <- list(
  `1` = train_by_pclass[["1"]],
  `2` = train_by_pclass[["2"]],
  `3` = train_by_pclass[["3"]]
)

# Step 3: Fit all 6 base models (3 GLM + 3 RF)
library(caret)

base_models <- list()
for (pc in c("1", "2", "3")) {
  # Logistic model
  base_models[[paste0("glm_", pc)]] <- train(
    formulas[[pc]],
    data = train_data[[pc]],
    method = "glm",
    family = "binomial",
    trControl = trainControl(classProbs = TRUE, summaryFunction = twoClassSummary),
    metric = "ROC"
  )
  
  # Random forest model
  base_models[[paste0("rf_", pc)]] <- train(
    formulas[[pc]],
    data = train_data[[pc]],
    method = "rf",
    trControl = trainControl(classProbs = TRUE, summaryFunction = twoClassSummary),
    metric = "ROC"
  )
}
# Build all 8 combinations using expand.grid
combos <- expand.grid("1" = c("glm", "rf"),
                      "2" = c("glm", "rf"),
                      "3" = c("glm", "rf"),
                      stringsAsFactors = FALSE)

# Generate named list of models for each combo
combo_models <- apply(combos, 1, function(row) {
  list(
    "1" = base_models[[paste0(row["1"], "_1")]],
    "2" = base_models[[paste0(row["2"], "_2")]],
    "3" = base_models[[paste0(row["3"], "_3")]]
  )
})
names(combo_models) <- apply(combos, 1, paste, collapse = "_")


###function ###

# --- Step 1: Define formulas for each class ---
fmla1 <- Survived ~ Sex + Age             # for class 1
fmla2 <- Survived ~ Sex + Age + Parch     # for class 2
fmla3 <- Survived ~ Sex + Age + SibSp     # for class 3

train_class1 <- train_by_pclass[["1"]]
train_class2 <- train_by_pclass[["2"]]
train_class3 <- train_by_pclass[["3"]]

# --- Step 2: Fit base models for each class (GLM + RF) ---
logit_class1 <- train(fmla1, data = train_class1, method = "glm", family = "binomial",
                      trControl = trainControl(classProbs = TRUE, summaryFunction = twoClassSummary), metric = "ROC")
logit_class2 <- train(fmla2, data = train_class2, method = "glm", family = "binomial",
                      trControl = trainControl(classProbs = TRUE, summaryFunction = twoClassSummary), metric = "ROC")
logit_class3 <- train(fmla3, data = train_class3, method = "glm", family = "binomial",
                      trControl = trainControl(classProbs = TRUE, summaryFunction = twoClassSummary), metric = "ROC")

rf_class1 <- train(fmla1, data = train_class1, method = "rf",
                   trControl = trainControl(classProbs = TRUE, summaryFunction = twoClassSummary), metric = "ROC")
rf_class2 <- train(fmla2, data = train_class2, method = "rf",
                   trControl = trainControl(classProbs = TRUE, summaryFunction = twoClassSummary), metric = "ROC")
rf_class3 <- train(fmla3, data = train_class3, method = "rf",
                   trControl = trainControl(classProbs = TRUE, summaryFunction = twoClassSummary), metric = "ROC")

# --- Step 3: Generate 8 model combinations ---
combos <- expand.grid(
  "1" = c("glm", "rf"),
  "2" = c("glm", "rf"),
  "3" = c("glm", "rf"),
  stringsAsFactors = FALSE
)
combo_names <- apply(combos, 1, paste, collapse = "_")

combo_models <- lapply(1:nrow(combos), function(i) {
  list(
    "1" = if (combos[i, "1"] == "glm") logit_class1 else rf_class1,
    "2" = if (combos[i, "2"] == "glm") logit_class2 else rf_class2,
    "3" = if (combos[i, "3"] == "glm") logit_class3 else rf_class3
  )
})
names(combo_models) <- combo_names

# --- Step 4: Prediction function you defined ---
predict_custom_combo <- function(test_df, model_list, combo_name = "custom") {
  test_split <- split(test_df, test_df$Pclass)
  
  preds <- lapply(names(model_list), function(pc) {
    model <- model_list[[pc]]
    newdata <- test_split[[pc]]
    if (is.null(newdata) || nrow(newdata) == 0) return(NULL)
    prob <- predict(model, newdata = newdata, type = "prob")[, "Yes"]
    data.frame(Combo = combo_name, Pclass = pc, prob = prob)
  })
  
  do.call(rbind, preds)
}

# --- Step 5: Predict all 8 combinations ---
combo_preds_list <- lapply(combo_names, function(name) {
  predict_custom_combo(test_df, combo_models[[name]], combo_name = name)
})

# --- Step 6: Combine into one table ---
all_combo_preds <- do.call(rbind, combo_preds_list)

# --- Step 7: View result ---
head(all_combo_preds)



##################

library(dplyr)

# Ensure test_df has PassengerId
test_df$PassengerId <- test_raw$PassengerId

# Create submission files
save_dir <- "D:/BU PHD/535/titanic/"

lapply(unique(all_combo_preds$Combo), function(combo_name) {
  preds <- all_combo_preds %>% 
    filter(Combo == combo_name) %>%
    group_by(Pclass) %>%
    mutate(RowID = row_number()) %>%
    ungroup()
  
  test_with_id <- test_df %>%
    mutate(Pclass = as.character(Pclass)) %>%
    group_by(Pclass) %>%
    mutate(RowID = row_number()) %>%
    ungroup()
  
  df_submit <- test_with_id %>%
    left_join(preds, by = c("Pclass", "RowID")) %>%
    mutate(Survived = ifelse(prob > 0.5, 1, 0)) %>%
    select(PassengerId, Survived)
  
  stopifnot(nrow(df_submit) == 418)
  
  write.csv(df_submit, file.path(save_dir, paste0("submission_combo_", combo_name, ".csv")), row.names = FALSE)
  
  return(df_submit)
}) -> all_combo_submissions

names(all_combo_submissions) <- unique(all_combo_preds$Combo)

##############gam ############

# If you haven't done this yet, read the data
train_data <- read.csv("D:/BU PHD/535/titanic/train.csv")

# Preprocess like before (select and convert variables)
library(dplyr)
train_data <- train_data %>%
  select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare) %>%
  mutate(
    Survived = factor(Survived, levels = c(0, 1), labels = c("No", "Yes")),
    Pclass = factor(Pclass),
    Sex = factor(Sex)
  ) %>%
  filter(!is.na(Age))  # remove NAs in Age for GAM
# Load the GAM package
library(mgcv)

# Fit GAM model with Age smoothed
gam_check <- gam(Survived ~ s(Age), data = train_data, family = binomial)
# Plot smooth function for Age
plot(gam_check, se = TRUE, col = "blue", shade = TRUE,
     xlab = "Age", ylab = "Effect on log-odds of Survival",
     main = "Smooth effect of Age")


################

library(caret)
library(randomForest)
train_data <- read.csv("D:/BU PHD/535/titanic/train.csv")



# Convert relevant variables to factor
train_data <- train_data %>%
  mutate(
    Survived = factor(Survived, levels = c(0, 1), labels = c("No", "Yes")),
    Pclass   = factor(Pclass),
    Sex      = factor(Sex)
  )

# Impute missing Age with median (overall median or group-specific if desired)
median_age <- median(train_data$Age, na.rm = TRUE)
train_data$Age[is.na(train_data$Age)] <- median_age

# Create AgeGroup
age_breaks <- c(-Inf, 15, 30, 50, Inf)
age_labels <- c("Child", "Young", "Adult", "Senior")

train_data$AgeGroup <- cut(train_data$Age, breaks = age_breaks, labels = age_labels)

# Split into Pclass-specific datasets
train_class1 <- filter(train_data, Pclass == 1)
train_class2 <- filter(train_data, Pclass == 2)
train_class3 <- filter(train_data, Pclass == 3)

# Set up control for training
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Class 1: GLM with Sex + AgeGroup
model_class1 <- train(Survived ~ Sex + AgeGroup, data = train_class1,
                      method = "glm", family = "binomial",
                      trControl = ctrl, metric = "ROC")

# Class 2: Random Forest with Sex + AgeGroup + Parch
model_class2 <- train(Survived ~ Sex + AgeGroup + Parch, data = train_class2,
                      method = "rf",
                      trControl = ctrl, metric = "ROC")

# Class 3: GLM with Sex + AgeGroup + SibSp
model_class3 <- train(Survived ~ Sex + AgeGroup + SibSp, data = train_class3,
                      method = "glm", family = "binomial",
                      trControl = ctrl, metric = "ROC")
library(caret)
library(randomForest)

# Set up control for training
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Class 1: GLM with Sex + AgeGroup
model_class1 <- train(Survived ~ Sex + AgeGroup, data = train_class1,
                      method = "glm", family = "binomial",
                      trControl = ctrl, metric = "ROC")

# Class 2: Random Forest with Sex + AgeGroup + Parch
model_class2 <- train(Survived ~ Sex + AgeGroup + Parch, data = train_class2,
                      method = "rf",
                      trControl = ctrl, metric = "ROC")

# Class 3: GLM with Sex + AgeGroup + SibSp
model_class3 <- train(Survived ~ Sex + AgeGroup + SibSp, data = train_class3,
                      method = "glm", family = "binomial",
                      trControl = ctrl, metric = "ROC")
custom_model_combo <- list(
  "1" = model_class1,
  "2" = model_class2,
  "3" = model_class3
)



# ===== Load test set =====
test_path <- "D:/BU PHD/535/titanic/test.csv"
test_data <- read.csv(test_path)

# ===== Impute missing Age using median and create AgeGroup =====
test_data$Age[is.na(test_data$Age)] <- median(test_data$Age, na.rm = TRUE)
test_data$AgeGroup <- cut(test_data$Age,
                          breaks = c(-Inf, 20, 40, 60, Inf),
                          labels = c("Child", "Young", "Adult", "Senior"))

# ===== Predict using class-specific models =====
test_split <- split(test_data, test_data$Pclass)

# Predict probabilities of "Yes" (i.e., Survived = 1)
prob1 <- predict(model_class1, newdata = test_split[["1"]], type = "prob")[, "Yes"]
prob2 <- predict(model_class2, newdata = test_split[["2"]], type = "prob")[, "Yes"]
prob3 <- predict(model_class3, newdata = test_split[["3"]], type = "prob")[, "Yes"]

# ===== Combine predictions =====
submission <- rbind(
  data.frame(PassengerId = test_split[["1"]]$PassengerId, Survived = ifelse(prob1 > 0.5, 1, 0)),
  data.frame(PassengerId = test_split[["2"]]$PassengerId, Survived = ifelse(prob2 > 0.5, 1, 0)),
  data.frame(PassengerId = test_split[["3"]]$PassengerId, Survived = ifelse(prob3 > 0.5, 1, 0))
)

save_dir <- "D:/BU PHD/535/titanic/"

write.csv(submission[order(submission$PassengerId), ], 
          file = file.path(save_dir, "submission_agegroup_model.csv"), 
          row.names = FALSE)







