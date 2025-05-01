# ©¤©¤©¤ 1. LIBRARIES & SEED ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
library(dplyr)    # data wrangling
library(tidyr)
library(caret)    # preprocessing (imputation, scaling)
library(caret)    # for createDataPartition()
library(glmnet)   # for cv.glmnet()

library(broom)
set.seed(42)

# ©¤©¤©¤ 2. LOAD & SELECT ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
df <- read.csv("D:/BU PHD/535/titanic/train.csv") %>%
  select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare)

# ©¤©¤©¤ 3. FACTOR CONVERSION ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
df <- df %>%
  mutate(
    Survived = factor(Survived,
                      levels = c(0, 1),
                      labels = c("No", "Yes")),
    Pclass   = factor(Pclass),
    Sex      = factor(Sex)
  )

# ©¤©¤©¤ 4. IMPUTE & SCALE NUMERIC PREDICTORS ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
num_preds <- c("Age", "SibSp", "Parch", "Fare")

# Build a preprocessor on numeric columns only
preproc <- preProcess(
  df[, num_preds],
  method = c("medianImpute", "center", "scale")
)

# Apply imputation, centering, and scaling
df[, num_preds] <- predict(preproc, df[, num_preds])

# ©¤©¤©¤ 5. RESULT ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
# 'df' is now cleaned, with:
#   ??? Survived as a two-level factor
#   ??? Pclass & Sex as factors
#   ??? Age, SibSp, Parch, Fare imputed (median) and standardized
head(df)

results_varsel <- list()

for (p in levels(df$Pclass)) {
  cat("\n=== Pclass", p, "===\n")
  dat   <- filter(df, Pclass == p)
  idx   <- createDataPartition(dat$Survived, p = 0.8, list = FALSE)
  train <- dat[idx, ]
  validation  <- dat[-idx, ]
  
  # 1) STEPWISE AIC
  null_mod <- glm(Survived ~ 1,
                  data   = train,
                  family = binomial)
  full_mod <- glm(Survived ~ Sex + Age + SibSp + Parch + Fare,
                  data   = train,
                  family = binomial)
  
  step_both <- step(null_mod,
                    scope     = list(lower = null_mod, upper = full_mod),
                    direction = "both",
                    trace     = FALSE)
  step_fwd  <- step(null_mod,
                    scope     = list(lower = null_mod, upper = full_mod),
                    direction = "forward",
                    trace     = FALSE)
  step_bwd  <- step(full_mod,
                    direction = "backward",
                    trace     = FALSE)
  # AIC table + plot for ¡°both¡±
  aic_both <- step_both$anova[, c("Step","AIC")]
  print(aic_both)
  plot(aic_both$AIC, type="b", xaxt="n",
       xlab="Step", ylab="AIC",
       main=paste("AIC (both) ¡ª Pclass", p))
  axis(1, at=seq_len(nrow(aic_both)), labels=aic_both$Step, las=2, cex.axis=0.7)
  
  # AIC table + plot for ¡°forward¡±
  aic_fwd <- step_fwd$anova[, c("Step","AIC")]
  print(aic_fwd)
  plot(aic_fwd$AIC, type="b", xaxt="n",
       xlab="Step", ylab="AIC",
       main=paste("AIC (forward) ¡ª Pclass", p))
  axis(1, at=seq_len(nrow(aic_fwd)), labels=aic_fwd$Step, las=2, cex.axis=0.7)
  
  # AIC table + plot for ¡°backward¡±
  aic_bwd <- step_bwd$anova[, c("Step","AIC")]
  print(aic_bwd)
  plot(aic_bwd$AIC, type="b", xaxt="n",
       xlab="Step", ylab="AIC",
       main=paste("AIC (backward) ¡ª Pclass", p))
  axis(1, at=seq_len(nrow(aic_bwd)), labels=aic_bwd$Step, las=2, cex.axis=0.7)
  
  ###
  print((coef(step_both)))
  
  cat("Stepwise (both)   :", names(coef(step_both))[-1], "\n\n")
  cat("Stepwise (forward):", names(coef(step_fwd))[-1], "\n\n")
  cat("Stepwise (backward):",names(coef(step_bwd))[-1], "\n\n")
  vars_both    <- names(coef(step_both))[-1]
  vars_forward <- names(coef(step_fwd))[-1]
  vars_backward<- names(coef(step_bwd))[-1]
  
  # 2) LASSO (¦Á = 1) via glmnet
  # build model matrix & numeric outcome
  x_train <- model.matrix(Survived ~ Sex + Age + SibSp + Parch + Fare,
                          data = train)[, -1]
  y_train <- as.numeric(train$Survived == "Yes")
  
  cv_lasso <- cv.glmnet(x_train, y_train,
                        family       = "binomial",
                        alpha        = 1,
                        nfolds       = 5,
                        type.measure = "auc")
  
  coef_lasso <- coef(cv_lasso, s = "lambda.min")
  
  # now index the very same object:
  # note that coef_lasso is a dgCMatrix, so we pull out the non-zero entries
  vars_lasso <- rownames(coef_lasso)[ which(coef_lasso[,1] != 0) ]
  # drop the intercept
  vars_lasso <- setdiff(vars_lasso, "(Intercept)")
  
  print(coef_lasso)
  cat("Mapped LASSO selected:", vars_lasso, "\n")
  
  
  
  cat("LASSO selected  :",  vars_lasso, "\n")
  
  
  # ©¤©¤©¤ CLEAN UP SEX DUMMY ¡ú FACTOR NAME ©¤
  clean_vars <- function(v) {
    v2 <- gsub("^Sex.*$", "Sex", v)
    unique(v2)
  }
  vars_both     <- clean_vars(vars_both)
  vars_forward  <- clean_vars(vars_forward)
  vars_backward <- clean_vars(vars_backward)
  vars_lasso    <- clean_vars(vars_lasso)
  
  
  # ©¤©¤©¤ 3) FIT LOGISTIC MODELS ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
  sel_list <- list(
    both     = vars_both,
    forward  = vars_forward,
    backward = vars_backward,
    lasso    = vars_lasso
  )
  
  glm_models <- list()    # <-- new
  
  for(m in names(sel_list)) {
    vars <- sel_list[[m]]
    if(length(vars)==0) next
    fmla    <- as.formula(paste("Survived ~", paste(vars, collapse=" + ")))
    glm_mod <- glm(fmla, data=train, family=binomial)
    
    # evaluate on hold©\out (you already have this)
    probs <- predict(glm_mod, newdata=validation, type="response")
    auc   <- roc(validation$Survived, probs, quiet=TRUE)$auc
    acc   <- mean((probs>0.5)==validation$Survived)
    cat(sprintf("[%s] vars: %-15s  AUC=%.3f  Acc=%.3f\n",
                m, paste(vars, collapse=","), auc, acc))
    
    glm_models[[m]] <- glm_mod   # <¡ª store it
  }
  
  # store everything, including the new list of glm_models
  results_varsel[[p]] <- list(
    step_both  = step_both,
    step_fwd   = step_fwd,
    step_bwd   = step_bwd,
    cv_lasso   = cv_lasso,
    glm_models = glm_models   # <¡ª added
  )
}


all_coefs <- bind_rows(
  lapply(names(results_varsel), function(pc) {
    gm <- results_varsel[[pc]]$glm_models
    bind_rows(
      lapply(names(gm), function(m) {
        tidy(gm[[m]]) %>%
          mutate(Pclass = pc,
                 method = m)
      })
    )
  })
)

# Reorder columns nicely:
all_coefs <- all_coefs %>%
  select(Pclass, method, term, estimate, std.error, statistic, p.value)




# 1) define the fixed ordering of methods
method_order <- c("both", "backward", "forward", "lasso")

# 2) collapse into one row per (Pclass,term,¡­) and collect methods
unique_coefs <- all_coefs %>%
  group_by(Pclass, term, estimate, std.error, statistic, p.value) %>%
  summarize(
    methods = paste(
      method_order[method_order %in% method],
      collapse = ","
    ),
    .groups = "drop"
  ) %>%
  # 3) now arrange the rows by the methods column
  mutate(methods = factor(methods, levels = unique(methods))) %>% 
  arrange(methods)

# 4) print everything
print(unique_coefs, n = nrow(unique_coefs))

# Pclass 1:
model_class1 <- results_varsel[["1"]]$glm_models$both

# Pclass 2:
model_class2 <- results_varsel[["2"]]$glm_models$both

# Pclass 3 (all four):
model_class3_both     <- results_varsel[["3"]]$glm_models$both
model_class3_forward  <- results_varsel[["3"]]$glm_models$forward
model_class3_backward <- results_varsel[["3"]]$glm_models$backward
model_class3_lasso    <- results_varsel[["3"]]$glm_models$lasso

# 1) Put them in a named list
models_list <- list(
  Pclass1           = model_class1,
  Pclass2           = model_class2,
  Pclass3_both      = model_class3_both,
  Pclass3_forward   = model_class3_forward,
  Pclass3_backward  = model_class3_backward,
  Pclass3_lasso     = model_class3_lasso
)

##########test################

test_raw <- read.csv("D:/BU PHD/535/titanic/test.csv")

test_df <- test_raw %>%
  select(PassengerId, Pclass, Sex, Age, SibSp, Parch, Fare) %>%
  mutate(
    # match factor levels exactly to training
    Pclass = factor(Pclass, levels = levels(df$Pclass)),
    Sex    = factor(Sex,    levels = levels(df$Sex))
  )

# Impute & standardize numeric predictors as on train
num_preds <- c("Age","SibSp","Parch","Fare")
test_df[, num_preds] <- predict(preproc, test_df[, num_preds])

# ©¤©¤©¤ 2) PREDICT SURVIVAL PROBABILITIES ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
test_preds <- test_df %>%
  mutate(
    Pred_1          = if_else(Pclass=="1",
                              predict(model_class1,    newdata = ., type="response"),
                              NA_real_),
    Pred_2          = if_else(Pclass=="2",
                              predict(model_class2,    newdata = ., type="response"),
                              NA_real_),
    Pred_3_both     = if_else(Pclass=="3",
                              predict(model_class3_both,    newdata = ., type="response"),
                              NA_real_),
    Pred_3_forward  = if_else(Pclass=="3",
                              predict(model_class3_forward, newdata = ., type="response"),
                              NA_real_),
    Pred_3_backward = if_else(Pclass=="3",
                              predict(model_class3_backward,newdata = ., type="response"),
                              NA_real_),
    Pred_3_lasso    = if_else(Pclass=="3",
                              predict(model_class3_lasso,   newdata = ., type="response"),
                              NA_real_)
  )

# ©¤©¤©¤ 3) EXTRACT & SAVE ONE RESULT PER Pclass3 MODEL ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
p3_methods <- c("both","forward","backward","lasso")
p3_results <- lapply(p3_methods, function(m) {
  prob_col <- paste0("Pred_3_", m)
  test_preds %>%
    filter(Pclass == "3") %>%
    transmute(
      PassengerId,
      Prob          = .data[[prob_col]],
      PredSurvived  = if_else(.data[[prob_col]] > 0.5, 1L, 0L)
    )
})
names(p3_results) <- p3_methods

# Assign to distinct variables if you like:
model3_both      <- p3_results[["both"]]
model3_forward   <- p3_results[["forward"]]
model3_backward  <- p3_results[["backward"]]
model3_lasso     <- p3_results[["lasso"]]

# ©¤©¤©¤ A) Path setup ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
test_path <- "D:/BU PHD/535/titanic/test.csv"
out_dir   <- dirname(test_path)   # same folder as test.csv

# ©¤©¤©¤ B) The four Pclass3 methods you want to compare ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
p3_methods <- c("both","forward","backward","lasso")

# ©¤©¤©¤ C) Loop over each method, build a full 418-row submission, and write ©¤©¤
for (m in p3_methods) {
  # the column in test_preds for this method
  p3_col <- paste0("Pred_3_", m)
  
  # build the submission: use Pred_1 for class1, Pred_2 for class2,
  # and the chosen Pred_3_<m> for class3
  submission <- test_preds %>%
    mutate(
      PredAll = case_when(
        Pclass == "1" ~ Pred_1,
        Pclass == "2" ~ Pred_2,
        Pclass == "3" ~ .data[[p3_col]]
      ),
      Survived = if_else(PredAll > 0.5, 1L, 0L)
    ) %>%
    select(PassengerId, Survived)
  
  # write out
  write.csv(
    submission,
    file      = file.path(out_dir, paste0("titanic_sub_class3_", m, ".csv")),
    row.names = FALSE
  )
}





