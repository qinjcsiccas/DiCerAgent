# =======================================================
# 脚本名称: predict_ABO4_tcf.R
# 作用: 供 Python 调用，加载 XGBoost 模型预测 ABO4 的 TCF 值
# =======================================================

# 1. 接收参数 (Input, Output, ModelPath)
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  stop("错误: 参数不足，需要3个参数: [输入CSV] [输出CSV] [模型路径]")
}

input_csv_path <- args[1]
output_csv_path <- args[2]
model_file_path <- args[3] # <--- 从 Python 接收模型路径

# 加载库
suppressMessages(library(xgboost))
suppressMessages(library(Matrix))

# 检查模型是否存在
if (!file.exists(model_file_path)) {
  stop(paste("R脚本报错: 找不到模型文件 ->", model_file_path))
}

# 2. 加载模型列表 (.rds)
model_list <- readRDS(model_file_path)

# 3. 读取 Python 传入的数据
# Python 传过来的 CSV 必须包含特征列 (如 sp, pm, cvd, bvs)
new_data <- read.csv(input_csv_path, stringsAsFactors = FALSE)

# 确保列名正确
feature_names <- c("sp", "pm", "cvd", "bvs")
if (!all(feature_names %in% colnames(new_data))) {
  stop(paste("缺少特征列! 需要:", feature_names, "实际:", colnames(new_data)))
}

# 4. 转换为 XGBoost 需要的 DMatrix，同时保留特征名称
data_matrix <- as.matrix(new_data[, feature_names])
colnames(data_matrix) <- feature_names
dinput <- xgb.DMatrix(data = data_matrix)

# 5. 批量预测 (Bagging)
n_models <- length(model_list)
n_samples <- nrow(new_data)
pred_matrix <- matrix(0, nrow = n_samples, ncol = n_models)

for (i in 1:n_models) {
  pred_matrix[, i] <- predict(model_list[[i]], dinput)
}

# 6. 计算统计结果 (均值和标准差)
final_mean <- rowMeans(pred_matrix)
final_sd <- apply(pred_matrix, 1, sd)

# 7. 导出结果
result_df <- data.frame(mean = final_mean, sd = final_sd)
write.csv(result_df, output_csv_path, row.names = FALSE)

# 输出成功信号
cat("ABO4_TCF_PREDICTION_COMPLETE")