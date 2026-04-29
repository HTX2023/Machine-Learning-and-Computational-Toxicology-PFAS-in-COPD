source("refer.ML.R")
############################################################
# Core machine-learning functions for binary classification
#
# This script provides a unified modeling interface for
# multiple machine-learning algorithms, including:
#   - Elastic Net / LASSO / Ridge
#   - Stepwise logistic regression
#   - Support vector machine
#   - Linear discriminant analysis
#   - glmBoost
#   - PLS-GLM
#   - Random forest
#   - Gradient boosting machine
#   - XGBoost
#   - Naive Bayes
#
# Main user-facing functions:
#   - RunML()
#   - ExtractVar()
#   - CalPredictScore()
#   - PredictClass()
#   - RunEval()
#   - scaleData()
#   - SimpleHeatmap()
#
# Required packages should be loaded in the main analysis
# script before sourcing this file.
############################################################


############################################################
# 1. Unified machine-learning wrapper
############################################################

RunML <- function(method, Train_set, Train_label, mode = "Model", classVar) {
  
  ##########################################################
  # RunML() provides a unified interface for all supported
  # machine-learning algorithms.
  #
  # Arguments:
  #   method:
  #     Name of the algorithm, e.g. "Lasso", "RF", "SVM",
  #     "Stepglm[backward]", or "Enet[alpha=0.4]".
  #
  #   Train_set:
  #     Training expression matrix or data frame.
  #     Rows represent samples; columns represent features.
  #
  #   Train_label:
  #     Data frame containing the binary outcome variable.
  #
  #   mode:
  #     "Model"    : build and return the fitted model.
  #     "Variable" : return selected features.
  #
  #   classVar:
  #     Name of the binary outcome variable in Train_label.
  #
  # Output:
  #   A fitted model object or a vector of selected variables.
  ##########################################################
  
  # Remove spaces in method names, e.g. "Enet [alpha=0.4]" -> "Enet[alpha=0.4]"
  method <- gsub(" ", "", method)
  
  # Extract algorithm name and parameters
  # Example:
  #   "Enet[alpha=0.4]" -> method_name = "Enet", method_param = "alpha=0.4"
  method_name <- gsub("(\\w+)\\[(.+)\\]", "\\1", method)
  method_param <- gsub("(\\w+)\\[(.+)\\]", "\\2", method)
  
  # Parse algorithm-specific parameters
  method_param <- switch(
    EXPR = method_name,
    "Enet" = list(alpha = as.numeric(gsub("alpha=", "", method_param))),
    "Stepglm" = list(direction = method_param),
    NULL
  )
  
  message(
    "Run ", method_name,
    " algorithm for ", mode,
    "; using ", ncol(Train_set), " variables."
  )
  
  # Construct input arguments for the selected algorithm
  args <- list(
    Train_set = Train_set,
    Train_label = Train_label,
    mode = mode,
    classVar = classVar
  )
  
  args <- c(args, method_param)
  
  # Call the corresponding algorithm-specific function
  obj <- do.call(
    what = paste0("Run", method_name),
    args = args
  )
  
  if (mode == "Variable") {
    message(length(obj), " variables retained.\n")
  } else {
    message("\n")
  }
  
  return(obj)
}


############################################################
# 2. Elastic Net, LASSO, and Ridge regression
############################################################

RunEnet <- function(Train_set, Train_label, mode, classVar, alpha) {
  
  ##########################################################
  # Elastic Net logistic regression.
  #
  # alpha = 1:
  #   LASSO regression.
  #
  # alpha = 0:
  #   Ridge regression.
  #
  # 0 < alpha < 1:
  #   Elastic Net regression.
  ##########################################################
  
  x <- as.matrix(Train_set)
  y <- Train_label[[classVar]]
  
  cv.fit <- cv.glmnet(
    x = x,
    y = y,
    family = "binomial",
    alpha = alpha,
    nfolds = 10
  )
  
  fit <- glmnet(
    x = x,
    y = y,
    family = "binomial",
    alpha = alpha,
    lambda = cv.fit$lambda.min
  )
  
  fit$subFeature <- colnames(Train_set)
  
  if (mode == "Model") return(fit)
  if (mode == "Variable") return(ExtractVar(fit))
}


RunLasso <- function(Train_set, Train_label, mode, classVar) {
  
  ##########################################################
  # LASSO logistic regression.
  # This is a special case of Elastic Net with alpha = 1.
  ##########################################################
  
  RunEnet(
    Train_set = Train_set,
    Train_label = Train_label,
    mode = mode,
    classVar = classVar,
    alpha = 1
  )
}


RunRidge <- function(Train_set, Train_label, mode, classVar) {
  
  ##########################################################
  # Ridge logistic regression.
  # This is a special case of Elastic Net with alpha = 0.
  ##########################################################
  
  RunEnet(
    Train_set = Train_set,
    Train_label = Train_label,
    mode = mode,
    classVar = classVar,
    alpha = 0
  )
}


############################################################
# 3. Stepwise logistic regression
############################################################

RunStepglm <- function(Train_set, Train_label, mode, classVar, direction, k_value = 2) {
  
  ##########################################################
  # Stepwise logistic regression based on AIC.
  #
  # direction:
  #   "both"     : bidirectional stepwise selection.
  #   "backward" : backward elimination.
  #   "forward"  : forward selection.
  #
  # k_value:
  #   Penalty parameter used by stepAIC().
  #   k = 2 corresponds to the standard AIC.
  ##########################################################
  
  data <- as.data.frame(Train_set)
  data[[classVar]] <- Train_label[[classVar]]
  
  fit <- stepAIC(
    glm(
      formula = as.formula(paste0(classVar, " ~ .")),
      family = "binomial",
      data = data
    ),
    direction = direction,
    trace = 0,
    k = k_value
  )
  
  fit$subFeature <- colnames(Train_set)
  
  if (mode == "Model") return(fit)
  if (mode == "Variable") return(ExtractVar(fit))
}


############################################################
# 4. Support vector machine
############################################################

RunSVM <- function(Train_set, Train_label, mode, classVar) {
  
  ##########################################################
  # Support vector machine classifier.
  #
  # Probability estimation is enabled to allow risk-score
  # calculation for ROC and AUC analyses.
  ##########################################################
  
  data <- as.data.frame(Train_set)
  data[[classVar]] <- as.factor(Train_label[[classVar]])
  
  fit <- svm(
    formula = as.formula(paste0(classVar, " ~ .")),
    data = data,
    probability = TRUE
  )
  
  fit$subFeature <- colnames(Train_set)
  
  if (mode == "Model") return(fit)
  if (mode == "Variable") return(ExtractVar(fit))
}


############################################################
# 5. Linear discriminant analysis
############################################################

RunLDA <- function(Train_set, Train_label, mode, classVar) {
  
  ##########################################################
  # Linear discriminant analysis implemented through caret.
  ##########################################################
  
  data <- as.data.frame(Train_set)
  data[[classVar]] <- as.factor(Train_label[[classVar]])
  
  fit <- train(
    as.formula(paste0(classVar, " ~ .")),
    data = data,
    method = "lda",
    trControl = trainControl(method = "cv", classProbs = TRUE)
  )
  
  fit$subFeature <- colnames(Train_set)
  
  if (mode == "Model") return(fit)
  if (mode == "Variable") return(ExtractVar(fit))
}


############################################################
# 6. glmBoost
############################################################

RunglmBoost <- function(Train_set, Train_label, mode, classVar) {
  
  ##########################################################
  # Boosted generalized linear model for binary classification.
  #
  # The optimal number of boosting iterations is estimated
  # using cross-validation risk.
  ##########################################################
  
  data <- cbind(as.data.frame(Train_set), Train_label[classVar])
  data[[classVar]] <- as.factor(data[[classVar]])
  
  fit <- glmboost(
    formula = as.formula(paste0(classVar, " ~ .")),
    data = data,
    family = Binomial()
  )
  
  cvm <- cvrisk(
    fit,
    papply = lapply,
    folds = cv(model.weights(fit), type = "kfold")
  )
  
  fit <- glmboost(
    formula = as.formula(paste0(classVar, " ~ .")),
    data = data,
    family = Binomial(),
    control = boost_control(mstop = max(mstop(cvm), 40))
  )
  
  fit$subFeature <- colnames(Train_set)
  
  if (mode == "Model") return(fit)
  if (mode == "Variable") return(ExtractVar(fit))
}


############################################################
# 7. Partial least squares generalized linear model
############################################################

RunplsRglm <- function(Train_set, Train_label, mode, classVar) {
  
  ##########################################################
  # Partial least squares generalized linear model.
  #
  # Logistic PLS-GLM is used for binary classification.
  # The number of components is restricted by the number of
  # available features.
  ##########################################################
  
  # Disable cross-validation when the number of variables is too small
  kfolds_to_use <- if (ncol(Train_set) < 3) 0 else 10
  
  if (kfolds_to_use > 0) {
    cv.plsRglm.res <- cv.plsRglm(
      formula = as.formula(paste0(classVar, " ~ .")),
      data = cbind(
        as.data.frame(Train_set),
        data.frame(Group = Train_label[[classVar]])
      ),
      nt = min(20, ncol(Train_set)),
      K = kfolds_to_use,
      verbose = FALSE
    )
  }
  
  fit <- plsRglm(
    Train_label[[classVar]],
    as.data.frame(Train_set),
    modele = "pls-glm-logistic",
    nt = min(20, ncol(Train_set)),
    verbose = FALSE,
    sparse = TRUE
  )
  
  fit$subFeature <- colnames(Train_set)
  
  if (mode == "Model") return(fit)
  if (mode == "Variable") return(ExtractVar(fit))
}


############################################################
# 8. Random forest
############################################################

RunRF <- function(Train_set, Train_label, mode, classVar) {
  
  ##########################################################
  # Random forest classifier based on randomForestSRC.
  #
  # Variable importance and variable selection are enabled.
  ##########################################################
  
  rf_nodesize <- 1
  
  # Ensure mtry does not exceed the number of available variables
  mtry_value <- min(5, ncol(Train_set))
  
  Train_label[[classVar]] <- as.factor(Train_label[[classVar]])
  
  fit <- rfsrc(
    formula = as.formula(paste0(classVar, " ~ .")),
    data = cbind(as.data.frame(Train_set), Train_label[classVar]),
    ntree = 1000,
    nodesize = rf_nodesize,
    mtry = mtry_value,
    importance = TRUE,
    proximity = TRUE,
    forest = TRUE
  )
  
  fit$subFeature <- colnames(Train_set)
  
  if (mode == "Model") return(fit)
  if (mode == "Variable") return(ExtractVar(fit))
}


############################################################
# 9. Gradient boosting machine
############################################################

RunGBM <- function(Train_set, Train_label, mode, classVar) {
  
  ##########################################################
  # Gradient boosting machine for binary classification.
  #
  # The optimal number of trees is determined by the minimum
  # cross-validation error.
  ##########################################################
  
  data <- as.data.frame(Train_set)
  data[[classVar]] <- Train_label[[classVar]]
  
  cv_folds <- min(10, nrow(data))
  
  fit <- gbm(
    formula = as.formula(paste0(classVar, " ~ .")),
    data = data,
    distribution = "bernoulli",
    n.trees = 10000,
    interaction.depth = 3,
    n.minobsinnode = 10,
    shrinkage = 0.001,
    cv.folds = cv_folds,
    n.cores = 6
  )
  
  best <- which.min(fit$cv.error)
  
  fit <- gbm(
    formula = as.formula(paste0(classVar, " ~ .")),
    data = data,
    distribution = "bernoulli",
    n.trees = best,
    interaction.depth = 3,
    n.minobsinnode = 10,
    shrinkage = 0.001,
    n.cores = 8
  )
  
  fit$subFeature <- colnames(Train_set)
  
  if (mode == "Model") return(fit)
  if (mode == "Variable") return(ExtractVar(fit))
}


############################################################
# 10. XGBoost
############################################################

RunXGBoost <- function(Train_set, Train_label, mode, classVar) {
  
  ##########################################################
  # XGBoost classifier for binary classification.
  #
  # Five-fold cross-validation is used to estimate the
  # optimal number of boosting rounds.
  ##########################################################
  
  x <- as.matrix(Train_set)
  y <- Train_label[[classVar]]
  
  indexes <- createFolds(y, k = 5, list = TRUE)
  
  CV <- unlist(lapply(indexes, function(pt) {
    
    dtrain <- xgb.DMatrix(
      data = x[-pt, , drop = FALSE],
      label = y[-pt]
    )
    
    dtest <- xgb.DMatrix(
      data = x[pt, , drop = FALSE],
      label = y[pt]
    )
    
    watchlist <- list(
      train = dtrain,
      test = dtest
    )
    
    bst <- xgb.train(
      data = dtrain,
      max.depth = 2,
      eta = 1,
      nthread = 2,
      nrounds = 10,
      watchlist = watchlist,
      objective = "binary:logistic",
      verbose = FALSE
    )
    
    which.min(bst$evaluation_log$test_logloss)
  }))
  
  nround <- as.numeric(names(which.max(table(CV))))
  
  if (is.na(nround) || nround < 1) {
    nround <- 1
  }
  
  fit <- xgboost(
    data = x,
    label = y,
    max.depth = 2,
    eta = 1,
    nthread = 2,
    nrounds = nround,
    objective = "binary:logistic",
    verbose = FALSE
  )
  
  fit$subFeature <- colnames(Train_set)
  
  if (mode == "Model") return(fit)
  if (mode == "Variable") return(ExtractVar(fit))
}


############################################################
# 11. Naive Bayes
############################################################

RunNaiveBayes <- function(Train_set, Train_label, mode, classVar) {
  
  ##########################################################
  # Naive Bayes classifier.
  ##########################################################
  
  data <- cbind(as.data.frame(Train_set), Train_label[classVar])
  data[[classVar]] <- as.factor(data[[classVar]])
  
  fit <- naiveBayes(
    as.formula(paste0(classVar, " ~ .")),
    data = data
  )
  
  fit$subFeature <- colnames(Train_set)
  
  if (mode == "Model") return(fit)
  if (mode == "Variable") return(ExtractVar(fit))
}


############################################################
# 12. Optional DRF function
############################################################

# The DRF algorithm is not used in the current binary
# classification workflow and is therefore retained only
# as a commented template.
#
# RunDRF <- function(Train_set, Train_label, mode, classVar) {
#   
#   Train_label <- data.frame(
#     "0" = as.numeric(Train_label == 0),
#     "1" = as.numeric(Train_label == 1)
#   )
#   
#   fit <- drf(
#     X = Train_set,
#     Y = Train_label,
#     compute.variable.importance = FALSE
#   )
#   
#   fit$subFeature <- colnames(Train_set)
#   
#   summary(
#     predict(
#       fit,
#       functional = "mean",
#       as.matrix(Train_set)
#     )$mean
#   )
#   
#   if (mode == "Model") return(fit)
#   if (mode == "Variable") return(ExtractVar(fit))
# }


############################################################
# 13. Utility function: suppress messages and console output
############################################################

quiet <- function(expr, messages = FALSE, cat = FALSE) {
  
  ##########################################################
  # Suppress unnecessary console output generated by some
  # model functions.
  #
  # Arguments:
  #   expr:
  #     R expression to evaluate.
  #
  #   messages:
  #     Whether messages should be retained.
  #
  #   cat:
  #     Whether console output should be retained.
  ##########################################################
  
  if (!cat) {
    sink(tempfile())
    on.exit(sink(), add = TRUE)
  }
  
  if (messages) {
    out <- eval.parent(substitute(expr))
  } else {
    out <- suppressMessages(eval.parent(substitute(expr)))
  }
  
  return(out)
}


############################################################
# 14. Utility function: standardization
############################################################

standarize.fun <- function(indata, centerFlag, scaleFlag) {
  
  ##########################################################
  # Standardize input data using base R scale().
  #
  # centerFlag:
  #   Whether to center variables.
  #
  # scaleFlag:
  #   Whether to scale variables to unit variance.
  ##########################################################
  
  scale(
    indata,
    center = centerFlag,
    scale = scaleFlag
  )
}


scaleData <- function(data, cohort = NULL, centerFlags = NULL, scaleFlags = NULL) {
  
  ##########################################################
  # Standardize expression data globally or by cohort.
  #
  # Arguments:
  #   data:
  #     Sample-by-feature expression matrix or data frame.
  #
  #   cohort:
  #     Optional cohort label for each sample.
  #     If provided, standardization is performed separately
  #     within each cohort.
  #
  #   centerFlags:
  #     Logical value or named logical vector indicating
  #     whether each cohort should be centered.
  #
  #   scaleFlags:
  #     Logical value or named logical vector indicating
  #     whether each cohort should be scaled.
  #
  # Output:
  #   Standardized matrix with the original sample order.
  ##########################################################
  
  samplename <- rownames(data)
  
  if (is.null(cohort)) {
    data <- list(data)
    names(data) <- "training"
  } else {
    data <- split(as.data.frame(data), cohort)
  }
  
  if (is.null(centerFlags)) {
    centerFlags <- FALSE
    message("No centerFlags found; set as FALSE.")
  }
  
  if (length(centerFlags) == 1) {
    centerFlags <- rep(centerFlags, length(data))
    message("Set centerFlags for all cohorts as ", unique(centerFlags), ".")
  }
  
  if (is.null(names(centerFlags))) {
    names(centerFlags) <- names(data)
    message("Match centerFlags with cohorts by order.\n")
  }
  
  if (is.null(scaleFlags)) {
    scaleFlags <- FALSE
    message("No scaleFlags found; set as FALSE.")
  }
  
  if (length(scaleFlags) == 1) {
    scaleFlags <- rep(scaleFlags, length(data))
    message("Set scaleFlags for all cohorts as ", unique(scaleFlags), ".")
  }
  
  if (is.null(names(scaleFlags))) {
    names(scaleFlags) <- names(data)
    message("Match scaleFlags with cohorts by order.\n")
  }
  
  centerFlags <- centerFlags[names(data)]
  scaleFlags <- scaleFlags[names(data)]
  
  outdata <- mapply(
    standarize.fun,
    indata = data,
    centerFlag = centerFlags,
    scaleFlag = scaleFlags,
    SIMPLIFY = FALSE
  )
  
  outdata <- do.call(rbind, outdata)
  outdata <- outdata[samplename, , drop = FALSE]
  
  return(outdata)
}


############################################################
# 15. Extract selected variables from model objects
############################################################

ExtractVar <- function(fit) {
  
  ##########################################################
  # Extract selected or retained variables from fitted models.
  #
  # For algorithms with embedded variable selection, variables
  # are selected according to non-zero coefficients or variable
  # importance.
  #
  # For algorithms without embedded feature selection, all
  # input features are retained by default.
  ##########################################################
  
  Feature <- quiet(switch(
    EXPR = class(fit)[1],
    
    # glmnet model:
    # variables with non-zero coefficients are retained.
    "lognet" = rownames(coef(fit))[which(coef(fit)[, 1] != 0)],
    
    # Stepwise logistic regression:
    # variables retained in the final regression model.
    "glm" = names(coef(fit)),
    
    # SVM:
    # no intrinsic variable selection; retain all input variables.
    "svm.formula" = fit$subFeature,
    
    # caret LDA:
    # no intrinsic variable selection; retain all input variables.
    "train" = fit$coefnames,
    
    # glmBoost:
    # variables with non-zero coefficients are retained.
    "glmboost" = names(coef(fit)[abs(coef(fit)) > 0]),
    
    # PLS-GLM:
    # variables with non-zero coefficients are retained.
    "plsRglmmodel" = rownames(fit$Coeffs)[fit$Coeffs != 0],
    
    # Random forest:
    # variables are selected using var.select().
    "rfsrc" = var.select(fit, verbose = FALSE)$topvars,
    
    # GBM:
    # variables with positive relative influence are retained.
    "gbm" = rownames(summary.gbm(fit, plotit = FALSE))[
      summary.gbm(fit, plotit = FALSE)$rel.inf > 0
    ],
    
    # XGBoost:
    # no intrinsic feature subset extraction here; retain all input variables.
    "xgb.Booster" = fit$subFeature,
    
    # Naive Bayes:
    # no intrinsic variable selection; retain all input variables.
    "naiveBayes" = fit$subFeature
  ))
  
  # Remove intercept terms if present
  Feature <- setdiff(
    Feature,
    c("(Intercept)", "Intercept")
  )
  
  return(Feature)
}


############################################################
# 16. Calculate model-based prediction scores
############################################################

CalPredictScore <- function(fit, new_data, type = "lp") {
  
  ##########################################################
  # Calculate prediction scores or risk probabilities for
  # new samples using a fitted model.
  #
  # Arguments:
  #   fit:
  #     Fitted model object.
  #
  #   new_data:
  #     Sample-by-feature matrix or data frame.
  #
  # Output:
  #   Numeric risk score vector for all samples.
  ##########################################################
  
  new_data <- new_data[, fit$subFeature, drop = FALSE]
  
  RS <- quiet(switch(
    EXPR = class(fit)[1],
    
    # glmnet logistic model
    "lognet" = predict(
      fit,
      type = "response",
      as.matrix(new_data)
    ),
    
    # logistic regression model
    "glm" = predict(
      fit,
      type = "response",
      as.data.frame(new_data)
    ),
    
    # support vector machine
    "svm.formula" = {
      pred <- predict(
        fit,
        as.data.frame(new_data),
        probability = TRUE
      )
      
      prob <- attr(pred, "probabilities")
      
      if (!is.null(prob)) {
        if ("1" %in% colnames(prob)) {
          prob[, "1"]
        } else {
          prob[, ncol(prob)]
        }
      } else {
        as.numeric(as.character(pred))
      }
    },
    
    # caret LDA model
    "train" = {
      prob <- predict(
        fit,
        new_data,
        type = "prob"
      )
      
      if ("1" %in% colnames(prob)) {
        prob[, "1"]
      } else {
        prob[, ncol(prob)]
      }
    },
    
    # glmBoost model
    "glmboost" = predict(
      fit,
      type = "response",
      as.data.frame(new_data)
    ),
    
    # PLS-GLM model
    "plsRglmmodel" = predict(
      fit,
      type = "response",
      as.data.frame(new_data)
    ),
    
    # random forest model
    "rfsrc" = {
      pred <- predict(
        fit,
        as.data.frame(new_data)
      )$predicted
      
      if ("1" %in% colnames(pred)) {
        pred[, "1"]
      } else {
        pred[, ncol(pred)]
      }
    },
    
    # GBM model
    "gbm" = predict(
      fit,
      type = "response",
      as.data.frame(new_data)
    ),
    
    # XGBoost model
    "xgb.Booster" = predict(
      fit,
      as.matrix(new_data)
    ),
    
    # Naive Bayes model
    "naiveBayes" = {
      pred <- predict(
        object = fit,
        type = "raw",
        newdata = new_data
      )
      
      if ("1" %in% colnames(pred)) {
        pred[, "1"]
      } else {
        pred[, ncol(pred)]
      }
    }
  ))
  
  RS <- as.numeric(as.vector(RS))
  names(RS) <- rownames(new_data)
  
  return(RS)
}


############################################################
# 17. Predict binary class labels
############################################################

PredictClass <- function(fit, new_data) {
  
  ##########################################################
  # Predict binary class labels for new samples using a
  # fitted machine-learning model.
  #
  # Output:
  #   Character vector of predicted class labels.
  ##########################################################
  
  new_data <- new_data[, fit$subFeature, drop = FALSE]
  
  label <- quiet(switch(
    EXPR = class(fit)[1],
    
    # glmnet logistic model
    "lognet" = predict(
      fit,
      type = "class",
      as.matrix(new_data)
    ),
    
    # logistic regression model
    "glm" = ifelse(
      predict(
        fit,
        type = "response",
        as.data.frame(new_data)
      ) > 0.5,
      "1",
      "0"
    ),
    
    # support vector machine
    "svm.formula" = predict(
      fit,
      as.data.frame(new_data),
      decision.values = TRUE
    ),
    
    # caret LDA model
    "train" = predict(
      fit,
      new_data,
      type = "raw"
    ),
    
    # glmBoost model
    "glmboost" = predict(
      fit,
      type = "class",
      as.data.frame(new_data)
    ),
    
    # PLS-GLM model
    "plsRglmmodel" = ifelse(
      predict(
        fit,
        type = "response",
        as.data.frame(new_data)
      ) > 0.5,
      "1",
      "0"
    ),
    
    # random forest model
    "rfsrc" = predict(
      fit,
      as.data.frame(new_data)
    )$class,
    
    # GBM model
    "gbm" = ifelse(
      predict(
        fit,
        type = "response",
        as.data.frame(new_data)
      ) > 0.5,
      "1",
      "0"
    ),
    
    # XGBoost model
    "xgb.Booster" = ifelse(
      predict(
        fit,
        as.matrix(new_data)
      ) > 0.5,
      "1",
      "0"
    ),
    
    # Naive Bayes model
    "naiveBayes" = predict(
      object = fit,
      type = "class",
      newdata = new_data
    )
  ))
  
  label <- as.character(as.vector(label))
  names(label) <- rownames(new_data)
  
  return(label)
}


############################################################
# 18. Evaluate model performance by AUC
############################################################

RunEval <- function(
    fit,
    Test_set = NULL,
    Test_label = NULL,
    Train_set = NULL,
    Train_label = NULL,
    Train_name = NULL,
    cohortVar = "Cohort",
    classVar
) {
  
  ##########################################################
  # Evaluate the predictive performance of a fitted model
  # using AUC in training and/or external validation cohorts.
  #
  # Arguments:
  #   fit:
  #     Fitted model object.
  #
  #   Test_set:
  #     External validation expression matrix.
  #
  #   Test_label:
  #     External validation phenotype data frame.
  #     It must contain both classVar and cohortVar.
  #
  #   Train_set:
  #     Optional training expression matrix.
  #
  #   Train_label:
  #     Optional training phenotype data frame.
  #
  #   Train_name:
  #     Name assigned to the training cohort.
  #
  #   cohortVar:
  #     Column name indicating cohort identity.
  #
  #   classVar:
  #     Column name indicating binary outcome.
  #
  # Output:
  #   AUC values for each cohort.
  ##########################################################
  
  if (!is.element(cohortVar, colnames(Test_label))) {
    stop(
      paste0(
        "There is no [",
        cohortVar,
        "] indicator. Please provide one additional cohort column."
      )
    )
  }
  
  # If training data are provided, evaluate training and
  # validation cohorts together.
  if ((!is.null(Train_set)) && (!is.null(Train_label))) {
    
    new_data <- rbind.data.frame(
      Train_set[, fit$subFeature, drop = FALSE],
      Test_set[, fit$subFeature, drop = FALSE]
    )
    
    if (!is.null(Train_name)) {
      Train_label$Cohort <- Train_name
    } else {
      Train_label$Cohort <- "Training"
    }
    
    colnames(Train_label)[ncol(Train_label)] <- cohortVar
    
    Test_label <- rbind.data.frame(
      Train_label[, c(cohortVar, classVar)],
      Test_label[, c(cohortVar, classVar)]
    )
    
    Test_label[, cohortVar] <- factor(
      Test_label[, cohortVar],
      levels = c(
        unique(Train_label[, cohortVar]),
        setdiff(
          unique(Test_label[, cohortVar]),
          unique(Train_label[, cohortVar])
        )
      )
    )
    
  } else {
    
    new_data <- Test_set[, fit$subFeature, drop = FALSE]
  }
  
  # Calculate risk scores
  RS <- suppressWarnings(
    CalPredictScore(
      fit = fit,
      new_data = new_data
    )
  )
  
  Predict.out <- Test_label
  Predict.out$RS <- as.vector(RS)
  
  # Split samples by cohort
  Predict.out <- split(
    x = Predict.out,
    f = Predict.out[, cohortVar]
  )
  
  # Calculate AUC for each cohort
  auc_values <- unlist(lapply(Predict.out, function(data) {
    
    as.numeric(
      auc(
        suppressMessages(
          roc(
            data[[classVar]],
            data$RS
          )
        )
      )
    )
  }))
  
  return(auc_values)
}


############################################################
# 19. AUC heatmap visualization
############################################################

SimpleHeatmap <- function(
    Cindex_mat,
    avg_Cindex,
    CohortCol,
    barCol,
    cellwidth = 1,
    cellheight = 0.5,
    cluster_columns,
    cluster_rows
) {
  
  ##########################################################
  # Draw a heatmap of AUC values across machine-learning
  # models and cohorts.
  #
  # Arguments:
  #   Cindex_mat:
  #     Matrix of AUC values.
  #     Rows are models; columns are cohorts.
  #
  #   avg_Cindex:
  #     Mean AUC value for each model.
  #
  #   CohortCol:
  #     Named color vector for cohorts.
  #
  #   barCol:
  #     Color of the right-side bar plot.
  #
  #   cellwidth, cellheight:
  #     Width and height of heatmap cells.
  #
  #   cluster_columns, cluster_rows:
  #     Whether to cluster columns or rows.
  ##########################################################
  
  col_ha <- columnAnnotation(
    "Cohort" = colnames(Cindex_mat),
    col = list("Cohort" = CohortCol),
    show_annotation_name = FALSE
  )
  
  row_ha <- rowAnnotation(
    bar = anno_barplot(
      avg_Cindex,
      bar_width = 0.8,
      border = FALSE,
      gp = grid::gpar(
        fill = barCol,
        col = NA
      ),
      add_numbers = TRUE,
      numbers_offset = grid::unit(-10, "mm"),
      axis_param = list(labels_rot = 0),
      numbers_gp = grid::gpar(
        fontsize = 9,
        col = "white"
      ),
      width = grid::unit(3, "cm")
    ),
    show_annotation_name = FALSE
  )
  
  Heatmap(
    as.matrix(Cindex_mat),
    name = "AUC",
    right_annotation = row_ha,
    top_annotation = col_ha,
    
    # Blue-white-red color scale for AUC visualization
    col = c("#4195C1", "#FFFFFF", "#CB5746"),
    
    # Add black borders to heatmap cells
    rect_gp = grid::gpar(
      col = "black",
      lwd = 1
    ),
    
    cluster_columns = cluster_columns,
    cluster_rows = cluster_rows,
    show_column_names = FALSE,
    show_row_names = TRUE,
    row_names_side = "left",
    
    width = grid::unit(
      cellwidth * ncol(Cindex_mat) + 2,
      "cm"
    ),
    
    height = grid::unit(
      cellheight * nrow(Cindex_mat),
      "cm"
    ),
    
    column_split = factor(
      colnames(Cindex_mat),
      levels = colnames(Cindex_mat)
    ),
    
    column_title = NULL,
    
    # Add AUC values to each heatmap cell
    cell_fun = function(j, i, x, y, w, h, col) {
      grid::grid.text(
        label = format(
          Cindex_mat[i, j],
          digits = 3,
          nsmall = 3
        ),
        x,
        y,
        gp = grid::gpar(fontsize = 10)
      )
    }
  )
}


############################################################
# End of refer.ML.R
############################################################