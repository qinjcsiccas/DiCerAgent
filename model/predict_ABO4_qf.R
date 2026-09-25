# =======================================================
# 脚本名称: predict_ABO4_qf.R
# 作用: 供 Python 调用，预测 ABO4 的 Qxf 值 (基于 SVM/Bagging)
# =======================================================

# 1. 接收命令行参数 (输入, 输出, 模型路径)
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  stop("错误: 参数不足，需要3个参数: [输入CSV] [输出CSV] [模型路径]")
}

input_path <- args[1]
output_path <- args[2]
model_path <- args[3]  # <--- 从 Python 接收模型路径

# 2. 加载库和模型
# 注意：如果你的模型是用其他包训练的(如 randomForest)，请在这里添加对应的 library
suppressMessages(library(e1071)) 

if(!file.exists(model_path)) {
  stop(paste("R脚本报错: 找不到模型文件 ->", model_path))
}

model_list <- readRDS(model_path)

# 3. 读取 Python 传过来的数据
# Python 传过来的 CSV 必须包含特征列 (如 vm, p, en)
new_data <- read.csv(input_path)

# 4. 批量预测
# 注意：SVM (e1071) 通常直接接受 data.frame，不需要像 XGBoost 那样转 DMatrix
n_models <- length(model_list)
n_samples <- nrow(new_data)
pred_matrix <- matrix(0, nrow = n_samples, ncol = n_models)

for (i in 1:n_models) {
  # 尝试预测，增加一点鲁棒性
  tryCatch({
    pred_matrix[, i] <- predict(model_list[[i]], new_data)
  }, error = function(e) {
    # 如果某个子模型预测失败，打印警告并填入 NA (或者均值)
    print(paste("Warning: Model", i, "failed:", e$message))
    pred_matrix[, i] <- NA
  })
}

# 5. 计算均值和标准差 (移除 NA 以防个别模型失败)
final_mean <- rowMeans(pred_matrix, na.rm = TRUE)
# 如果只有一个模型，sd 会报错，这里做一个判断
if (n_models > 1) {
  final_sd <- apply(pred_matrix, 1, sd, na.rm = TRUE)
} else {
  final_sd <- 0
}

# 6. 输出结果
result_df <- data.frame(mean = final_mean, sd = final_sd)
write.csv(result_df, output_path, row.names = FALSE)

# 输出成功信号
cat("QXF_PREDICTION_COMPLETE")