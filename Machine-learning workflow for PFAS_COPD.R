############################################################
# Machine-learning workflow for COPD classification
# using GEO lung tissue transcriptomic datasets
#
# Training dataset:
#   - GSE151052
#
# External validation datasets:
#   - GSE76925
#   - GSE38974
#
# Main workflow:
#   1. Extract hub-gene expression matrices from each GEO dataset
#   2. Construct training and validation datasets
#   3. Harmonize feature genes across cohorts
#   4. Perform machine-learning model construction
#   5. Evaluate model performance using AUC
#   6. Visualize model-level and gene-level ROC curves
############################################################


############################################################
# 0. Clear workspace
############################################################

rm(list = ls())


############################################################
# 1. Define helper function for preparing GEO datasets
############################################################

prepare_geo_dataset <- function(
    rdata_path,
    dataset_prefix,
    gene_file = "机器前hub基因.csv",
    output_file
) {
  ##########################################################
  # This function loads one preprocessed GEO expression file,
  # extracts candidate hub genes, transposes the expression
  # matrix into sample-by-gene format, appends cohort and
  # phenotype labels to sample IDs, and exports the final
  # machine-learning-ready dataset.
  #
  # Required objects in the loaded RData file:
  #   - exp: gene-by-sample expression matrix
  #   - Group or clinical$Group: phenotype labels
  #
  # Output:
  #   - A CSV file with samples as rows, genes as columns,
  #     and a binary Group column:
  #       1 = COPD
  #       0 = non-COPD / Normal / control
  ##########################################################
  
  rm(list = setdiff(ls(envir = .GlobalEnv), c("prepare_geo_dataset")))
  
  # Load the preprocessed GEO dataset
  load(rdata_path)
  
  # Check whether the required expression matrix exists
  if (!exists("exp")) {
    stop("The object 'exp' was not found in the loaded RData file.")
  }
  
  # Display basic information for quality checking
  cat("\n==============================\n")
  cat("Processing dataset:", dataset_prefix, "\n")
  cat("==============================\n")
  cat("Expression matrix dimension:\n")
  print(dim(exp))
  
  if (exists("Group")) {
    cat("Group distribution:\n")
    print(table(Group))
  }
  
  # Read candidate hub genes
  gene_data <- read.csv(
    gene_file,
    header = TRUE,
    stringsAsFactors = FALSE
  )
  
  # Extract gene symbols from the first column
  genes <- gene_data[, 1]
  
  # Remove missing values, empty strings, and duplicated genes
  genes <- genes[!is.na(genes) & genes != ""]
  genes <- unique(genes)
  
  # Keep only genes present in the expression matrix
  selected_genes <- intersect(genes, rownames(exp))
  
  if (length(selected_genes) == 0) {
    stop("None of the candidate genes were found in the expression matrix.")
  }
  
  cat("Number of candidate genes provided:", length(genes), "\n")
  cat("Number of genes retained in this dataset:", length(selected_genes), "\n")
  
  # Subset the expression matrix and transpose it
  # Original format: genes x samples
  # New format: samples x genes
  exp_sub <- exp[selected_genes, , drop = FALSE]
  exp_sub <- t(exp_sub)
  
  # Obtain phenotype labels
  if (exists("clinical") && "Group" %in% colnames(clinical)) {
    group_vector <- as.character(clinical$Group)
  } else if (exists("Group")) {
    group_vector <- as.character(Group)
  } else {
    stop("Neither 'clinical$Group' nor 'Group' was found for phenotype annotation.")
  }
  
  # Check whether the number of phenotype labels matches samples
  if (length(group_vector) != nrow(exp_sub)) {
    stop("The number of phenotype labels does not match the number of samples.")
  }
  
  # Preserve the original sample IDs
  original_sample_ids <- rownames(exp_sub)
  
  # Add dataset prefix and phenotype label to sample IDs
  rownames(exp_sub) <- paste0(
    dataset_prefix,
    "_",
    original_sample_ids,
    "_",
    group_vector
  )
  
  # Convert matrix to data frame
  exp_df <- as.data.frame(exp_sub)
  
  # Create binary phenotype label:
  # COPD samples are coded as 1; all other samples are coded as 0.
  exp_df$Group <- ifelse(
    grepl("COPD", group_vector, ignore.case = TRUE),
    1,
    0
  )
  
  # Export processed dataset
  write.csv(exp_df, file = output_file, row.names = TRUE)
  
  cat("Processed dataset saved to:", output_file, "\n")
  cat("Final dataset dimension:\n")
  print(dim(exp_df))
  
  return(exp_df)
}


############################################################
# 2. Prepare training dataset: GSE151052
############################################################

train_gse151052 <- prepare_geo_dataset(
  rdata_path = "/home/hzy-sc1/AAAAA生信分析全流程/芯片数据-pipeline/GEO的公共数据/GSE151052——COPD人肺组织——151052外验证已处理/GSE151052整理.RData",
  dataset_prefix = "GSE151052",
  gene_file = "机器前hub基因.csv",
  output_file = "train_data.csv"
)


############################################################
# 3. Prepare external validation dataset 1: GSE76925
############################################################

test_gse76925 <- prepare_geo_dataset(
  rdata_path = "/home/hzy-sc1/AAAAA生信分析全流程/芯片数据-pipeline/GEO的公共数据/GSE76925——COPD人肺组织——外验证已处理/GSE76925整理.Rdata",
  dataset_prefix = "GSE76925",
  gene_file = "机器前hub基因.csv",
  output_file = "test1_data.csv"
)


############################################################
# 4. Prepare external validation dataset 2: GSE38974
############################################################

test_gse38974 <- prepare_geo_dataset(
  rdata_path = "/home/hzy-sc1/AAAAA生信分析全流程/芯片数据-pipeline/GEO的公共数据/GSE38974——COPD肺组织——外验证已处理/GSE38974外验证.RData",
  dataset_prefix = "GSE38974",
  gene_file = "机器前hub基因.csv",
  output_file = "test2_data.csv"
)


############################################################
# 5. Merge training and validation datasets
############################################################

# Read processed training and validation datasets
train <- read.table(
  "train_data.csv",
  header = TRUE,
  row.names = 1,
  sep = ",",
  check.names = FALSE
)

test1 <- read.table(
  "test1_data.csv",
  header = TRUE,
  row.names = 1,
  sep = ",",
  check.names = FALSE
)

test2 <- read.table(
  "test2_data.csv",
  header = TRUE,
  row.names = 1,
  sep = ",",
  check.names = FALSE
)

# Check column names
cat("Training dataset columns:\n")
print(colnames(train))

cat("Validation dataset 1 columns:\n")
print(colnames(test1))

cat("Validation dataset 2 columns:\n")
print(colnames(test2))

# Harmonize columns across all datasets
# Only features shared by all datasets are retained.
common_columns <- Reduce(
  intersect,
  list(colnames(train), colnames(test1), colnames(test2))
)

if (!"Group" %in% common_columns) {
  stop("The binary phenotype column 'Group' is missing from the merged feature set.")
}

train <- train[, common_columns, drop = FALSE]
test1 <- test1[, common_columns, drop = FALSE]
test2 <- test2[, common_columns, drop = FALSE]

# Merge two validation datasets
test <- rbind(test1, test2)

# Export the harmonized training and validation datasets
write.csv(test, file = "test_merge_data.csv", row.names = TRUE)
write.csv(train, file = "train_merge_data.csv", row.names = TRUE)

cat("Merged training dataset dimension:\n")
print(dim(train))

cat("Merged validation dataset dimension:\n")
print(dim(test))


############################################################
# 6. Load required R packages for machine-learning analysis
############################################################

library(openxlsx)
library(seqinr)
library(plyr)
library(randomForestSRC)
library(glmnet)
library(plsRglm)
library(gbm)
library(caret)
library(mboost)
library(e1071)
library(BART)
library(MASS)
library(snowfall)
library(xgboost)
library(ComplexHeatmap)
library(RColorBrewer)
library(pROC)
library(UpSetR)

# Source customized machine-learning functions.
# The file refer.ML.R should contain user-defined functions such as:
#   - RunML()
#   - ExtractVar()
#   - CalPredictScore()
#   - PredictClass()
#   - RunEval()
#   - scaleData()
#   - SimpleHeatmap()
source("refer.ML.R")


############################################################
# 7. Read training and validation data for model construction
############################################################

# Read the harmonized training dataset
train_data_raw <- read.table(
  "train_merge_data.csv",
  header = TRUE,
  row.names = 1,
  sep = ",",
  check.names = FALSE
)

# Separate expression features and phenotype labels
train_features <- train_data_raw[, -ncol(train_data_raw), drop = FALSE]
train_labels <- train_data_raw[, ncol(train_data_raw), drop = FALSE]

# Read the harmonized validation dataset
test_data_raw <- read.table(
  "test_merge_data.csv",
  header = TRUE,
  row.names = 1,
  sep = ",",
  check.names = FALSE
)

# Separate expression features and phenotype labels
test_features <- test_data_raw[, -ncol(test_data_raw), drop = FALSE]
test_labels <- test_data_raw[, ncol(test_data_raw), drop = FALSE]

# Extract cohort information from sample IDs
# Expected sample ID format:
#   GSEID_originalSampleID_Group
test_labels$Cohort <- gsub(
  "(.+)\\_(.+)\\_(.+)",
  "\\1",
  rownames(test_data_raw)
)


############################################################
# 8. Harmonize feature genes between training and validation sets
############################################################

# Retain only common genes between training and validation datasets
common_features <- intersect(
  colnames(train_features),
  colnames(test_features)
)

if (length(common_features) == 0) {
  stop("No common feature genes were found between the training and validation datasets.")
}

cat("Number of common feature genes:", length(common_features), "\n")

# Convert expression data to matrices
train_data <- as.matrix(train_features[, common_features, drop = FALSE])
test_data <- as.matrix(test_features[, common_features, drop = FALSE])

# Standardize training data
train_data <- scaleData(
  train_data,
  centerFlags = TRUE,
  scaleFlags = TRUE
)

# Standardize validation data by cohort
test_data <- scaleData(
  test_data,
  cohort = test_labels$Cohort,
  centerFlags = TRUE,
  scaleFlags = TRUE
)


############################################################
# 9. Define machine-learning settings
############################################################

# Read the list of machine-learning method combinations.
# The file should contain one column named "x",
# with each row representing one model or model combination.
methods <- read.table(
  "113_ML_methods.txt",
  header = TRUE,
  sep = "\t",
  check.names = FALSE
)

methods <- methods$x

# Define the classification outcome variable
classVar <- "Group"

# Minimum number of selected variables required for a valid model
min.selected.var <- 1

# Candidate variables
Variable <- colnames(train_features)

# Methods used for preliminary feature selection
preTrain.method <- c(
  "Lasso",
  "glmBoost",
  "RF",
  "Stepglm[both]",
  "Stepglm[backward]"
)


############################################################
# 10. Preliminary feature selection using multiple algorithms
############################################################

# Initialize list to store genes selected by each algorithm
preTrain.var <- list()

# Initialize list to store running time for each algorithm
time.list <- list()

set.seed(123)

for (method in preTrain.method) {
  
  cat("\nRunning preliminary feature selection using:", method, "\n")
  
  time.taken <- system.time({
    preTrain.var[[method]] <- RunML(
      method = method,
      Train_set = train_data,
      Train_label = train_labels,
      mode = "Variable",
      classVar = classVar
    )
  })
  
  # Store elapsed running time in seconds
  time.list[[method]] <- time.taken[["elapsed"]]
  
  cat(sprintf(
    "Method [%s] completed in %.2f seconds.\n",
    method,
    time.taken[["elapsed"]]
  ))
}

# The "simple" strategy uses all genes without preliminary feature selection
preTrain.var[["simple"]] <- colnames(train_data)


############################################################
# 11. Visualize overlap among genes selected by different algorithms
############################################################

# Extract gene lists from each feature-selection algorithm
gene_lists <- list(
  Lasso = preTrain.var$Lasso,
  glmBoost = preTrain.var$glmBoost,
  RF = preTrain.var$RF,
  Step_both = preTrain.var$`Stepglm[both]`,
  Step_back = preTrain.var$`Stepglm[backward]`
)

# Convert gene lists into UpSetR-compatible format
upset_data <- fromList(gene_lists)

# Define colors for the UpSet plot
allcolour <- c(
  "#DF0A1F",
  "#1C5BA7",
  "#019E73",
  "#ED621B",
  "#E477C1"
)

# Generate UpSet plot showing overlap of selected genes
pdf("gene_intersection_upset.pdf", width = 10, height = 6)

upset(
  upset_data,
  sets = names(gene_lists),
  order.by = "freq",
  text.scale = 1.2,
  matrix.color = allcolour,
  mainbar.y.label = "Gene number intersected",
  sets.x.label = "Gene number selected"
)

dev.off()

# Identify genes selected by all feature-selection algorithms
core_genes <- Reduce(intersect, gene_lists)

# Export core genes
write.table(
  core_genes,
  "core_genes.txt",
  quote = FALSE,
  row.names = FALSE,
  col.names = FALSE
)


############################################################
# 12. Construct machine-learning models
############################################################

# Initialize list to store final machine-learning models
model <- list()

set.seed(123)

# Backup the full training expression matrix
Train_set_bk <- train_data

for (method in methods) {
  
  cat(match(method, methods), ":", method, "\n")
  
  method_name <- method
  
  # Split combined method name into:
  #   feature-selection method + modeling method
  method_split <- strsplit(method, "\\+")[[1]]
  
  # If only one modeling method is provided,
  # use all genes as the input feature set.
  if (length(method_split) == 1) {
    method_split <- c("simple", method_split)
  }
  
  # Retrieve genes selected by the first-stage method
  selected_variables <- preTrain.var[[method_split[1]]]
  
  if (is.null(selected_variables) || length(selected_variables) == 0) {
    cat("No variables selected by:", method_split[1], "\n")
    next
  }
  
  # Subset the training data using selected genes
  train_data_sub <- Train_set_bk[, selected_variables, drop = FALSE]
  
  # Build machine-learning model
  model[[method_name]] <- RunML(
    method = method_split[2],
    Train_set = train_data_sub,
    Train_label = train_labels,
    mode = "Model",
    classVar = classVar
  )
  
  # Remove invalid models with too few selected variables
  if (length(ExtractVar(model[[method_name]])) <= min.selected.var) {
    model[[method_name]] <- NULL
  }
}

# Restore the full training expression matrix
train_data <- Train_set_bk
rm(Train_set_bk)

# Save all valid machine-learning models
saveRDS(model, "model.MLmodel.rds")


############################################################
# 13. Calculate prediction scores for each sample
############################################################

# Load saved machine-learning models
model <- readRDS("model.MLmodel.rds")

# Extract names of valid models
methodsValid <- names(model)

# Calculate prediction scores for training and validation samples
RS_list <- list()

for (method in methodsValid) {
  RS_list[[method]] <- CalPredictScore(
    fit = model[[method]],
    new_data = rbind.data.frame(train_data, test_data)
  )
}

# Convert prediction scores into a matrix
riskTab <- as.data.frame(t(do.call(rbind, RS_list)))
riskTab <- cbind(id = rownames(riskTab), riskTab)

# Export prediction score matrix
write.table(
  riskTab,
  "model.riskMatrix.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)


############################################################
# 14. Predict sample classes using each machine-learning model
############################################################

Class_list <- list()

for (method in methodsValid) {
  Class_list[[method]] <- PredictClass(
    fit = model[[method]],
    new_data = rbind.data.frame(train_data, test_data)
  )
}

# Convert predicted classes into a matrix
Class_mat <- as.data.frame(t(do.call(rbind, Class_list)))

# Export predicted class matrix
classTab <- cbind(id = rownames(Class_mat), Class_mat)

write.table(
  classTab,
  "model.classMatrix.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)


############################################################
# 15. Extract feature genes used by each valid model
############################################################

fea_list <- list()

for (method in methodsValid) {
  fea_list[[method]] <- ExtractVar(model[[method]])
}

fea_df <- lapply(model, function(fit) {
  data.frame(ExtractVar(fit))
})

fea_df <- do.call(rbind, fea_df)

# Extract algorithm name from row names
fea_df$algorithm <- gsub(
  "(.+)\\.(.+$)",
  "\\1",
  rownames(fea_df)
)

colnames(fea_df)[1] <- "features"

# Export model-selected genes
write.table(
  fea_df,
  file = "model.genes.txt",
  sep = "\t",
  row.names = FALSE,
  col.names = TRUE,
  quote = FALSE
)


############################################################
# 16. Evaluate model performance using AUC
############################################################

AUC_list <- list()

for (method in methodsValid) {
  
  AUC_list[[method]] <- RunEval(
    fit = model[[method]],
    Test_set = test_data,
    Test_label = test_labels,
    Train_set = train_data,
    Train_label = train_labels,
    Train_name = "Train",
    cohortVar = "Cohort",
    classVar = classVar
  )
}

# Combine AUC values from all models
AUC_mat <- do.call(rbind, AUC_list)

aucTab <- cbind(
  Method = rownames(AUC_mat),
  AUC_mat
)

# Export AUC matrix
write.table(
  aucTab,
  "model.AUCmatrix.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)


############################################################
# 17. Draw AUC heatmap for all machine-learning models
############################################################

# Read AUC matrix
AUC_mat <- read.table(
  "model.AUCmatrix.txt",
  header = TRUE,
  sep = "\t",
  check.names = FALSE,
  row.names = 1,
  stringsAsFactors = FALSE
)

# Rank models according to mean AUC across cohorts
avg_AUC <- apply(AUC_mat, 1, mean)
avg_AUC <- sort(avg_AUC, decreasing = TRUE)

AUC_mat <- AUC_mat[names(avg_AUC), , drop = FALSE]

# Extract feature genes from the best-performing model
best_model_name <- rownames(AUC_mat)[1]
fea_sel <- fea_list[[best_model_name]]

cat("Best-performing model:", best_model_name, "\n")
cat("Number of genes selected by the best model:", length(fea_sel), "\n")

# Format mean AUC values
avg_AUC <- as.numeric(format(avg_AUC, digits = 3, nsmall = 3))

# Define cohort colors
CohortCol <- brewer.pal(
  n = ncol(AUC_mat),
  name = "Paired"
)

names(CohortCol) <- colnames(AUC_mat)

# Set heatmap cell size
cellwidth <- 1
cellheight <- 0.5

# Draw AUC heatmap
hm <- SimpleHeatmap(
  Cindex_mat = AUC_mat,
  avg_Cindex = avg_AUC,
  CohortCol = CohortCol,
  barCol = "steelblue",
  cellwidth = cellwidth,
  cellheight = cellheight,
  cluster_columns = FALSE,
  cluster_rows = FALSE
)

# Export AUC heatmap
pdf(
  file = "ML_AUC_heatmap.pdf",
  width = cellwidth * ncol(AUC_mat) + 6,
  height = cellheight * nrow(AUC_mat) * 0.45
)

draw(
  hm,
  heatmap_legend_side = "right",
  annotation_legend_side = "right"
)

dev.off()


############################################################
# 18. Draw ROC curves for the selected best model in each cohort
############################################################

library(pROC)

# Input prediction score matrix
rsFile <- "model.riskMatrix.txt"

# Select the best-performing model.
# Please modify this value according to the AUC heatmap if needed.
method <- "Stepglm[backward]+plsRglm"

# Read model prediction scores
riskRT <- read.table(
  rsFile,
  header = TRUE,
  sep = "\t",
  check.names = FALSE,
  row.names = 1
)

# Extract cohort ID from sample names
CohortID <- gsub(
  "(.*)\\_(.*)\\_(.*)",
  "\\1",
  rownames(riskRT)
)

CohortID <- gsub(
  "(.*)\\.(.*)",
  "\\1",
  CohortID
)

riskRT$Cohort <- CohortID

# Check whether the selected model exists in the risk matrix
if (!method %in% colnames(riskRT)) {
  stop("The selected model was not found in the prediction score matrix. Please check the method name.")
}

# Draw ROC curve for each cohort
for (Cohort in unique(riskRT$Cohort)) {
  
  # Extract samples from the current cohort
  rt <- riskRT[riskRT$Cohort == Cohort, , drop = FALSE]
  
  # Extract phenotype labels from sample names
  # Normal samples are coded as 0; non-Normal samples are coded as 1.
  y <- gsub(".*_(.*)$", "\\1", rownames(rt))
  y <- ifelse(y == "Normal", 0, 1)
  
  # Ensure that both phenotype classes are present
  if (length(unique(y)) != 2) {
    cat(
      "Error: y must contain exactly two unique values. Skipping cohort:",
      Cohort,
      "\n"
    )
    next
  }
  
  # Calculate ROC curve
  roc1 <- roc(
    y,
    as.numeric(rt[, method])
  )
  
  # Calculate 95% confidence interval of AUC using bootstrap
  ci1 <- ci.auc(
    roc1,
    method = "bootstrap"
  )
  
  ciVec <- as.numeric(ci1)
  
  # Export ROC curve
  pdf(
    file = paste0("ROC.", Cohort, ".pdf"),
    width = 5,
    height = 4.75
  )
  
  plot(
    roc1,
    print.auc = TRUE,
    col = "red",
    legacy.axes = TRUE,
    main = Cohort
  )
  
  text(
    0.39,
    0.43,
    paste0(
      "95% CI: ",
      sprintf("%.03f", ciVec[1]),
      "-",
      sprintf("%.03f", ciVec[3])
    ),
    col = "red"
  )
  
  dev.off()
}


############################################################
# 19. Draw gene-level ROC curves for genes selected by the best model
############################################################

library(glmnet)
library(pROC)

# Input files
geneFile <- "model.genes.txt"

# Read training dataset
rt <- read.table(
  "train_merge_data.csv",
  header = TRUE,
  row.names = 1,
  sep = ",",
  check.names = FALSE
)

# Transpose expression matrix:
# Original format: samples x genes
# New format: genes x samples
rt <- t(rt)

# Extract phenotype labels from sample names
y <- gsub(".*_(.*)$", "\\1", colnames(rt))

# Normal samples are coded as 0; non-Normal samples are coded as 1.
y <- ifelse(y == "Normal", 0, 1)

# Read model-selected gene list
geneRT <- read.table(
  geneFile,
  header = TRUE,
  sep = "\t",
  check.names = FALSE
)

# Extract genes selected by the specified model
geneRT <- geneRT$features[
  geneRT$algorithm == "Stepglm[backward]+plsRglm"
]

# Remove duplicated genes
geneRT <- unique(geneRT)

# Check whether selected genes are present in the expression matrix
geneRT <- intersect(geneRT, rownames(rt))

if (length(geneRT) == 0) {
  stop("No selected model genes were found in the training expression matrix.")
}

# Define colors for gene-level ROC curves
bioCol <- rainbow(
  length(geneRT),
  s = 0.9,
  v = 0.9
)

# Draw ROC curves for each selected gene
aucText <- c()
k <- 0

for (x in as.vector(geneRT)) {
  
  k <- k + 1
  
  # Calculate gene-level ROC curve
  roc1 <- roc(
    y,
    as.numeric(rt[x, ])
  )
  
  if (k == 1) {
    
    pdf(
      file = "ROC.genes.pdf",
      width = 9,
      height = 9
    )
    
    plot(
      roc1,
      print.auc = FALSE,
      col = bioCol[k],
      legacy.axes = TRUE,
      main = "",
      lwd = 3
    )
    
    aucText <- c(
      aucText,
      paste0(
        x,
        ", AUC=",
        sprintf("%.3f", roc1$auc[1])
      )
    )
    
  } else {
    
    plot(
      roc1,
      print.auc = FALSE,
      col = bioCol[k],
      legacy.axes = TRUE,
      main = "",
      lwd = 3,
      add = TRUE
    )
    
    aucText <- c(
      aucText,
      paste0(
        x,
        ", AUC=",
        sprintf("%.3f", roc1$auc[1])
      )
    )
  }
}

# Add legend showing AUC values for each gene
legend(
  "bottomright",
  aucText,
  lwd = 3,
  bty = "n",
  cex = 0.8,
  col = bioCol[seq_along(geneRT)],
  inset = c(0.05, 0)
)

dev.off()


############################################################
# End of script
############################################################