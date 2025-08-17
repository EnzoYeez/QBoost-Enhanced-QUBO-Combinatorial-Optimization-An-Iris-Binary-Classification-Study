import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

# === 1. 读取数据 ===
train_df = pd.read_csv("iris_train.csv")
test_df = pd.read_csv("iris_test.csv")

X_train = train_df.drop(columns=["label"]).values
y_train = np.where(train_df["label"].values == 0, -1, 1)
X_test = test_df.drop(columns=["label"]).values
y_test = np.where(test_df["label"].values == 0, -1, 1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === 2. 枚举所有1~4特征组合 ===
num_features = X_train.shape[1]
results = []

for k in range(1, num_features + 1):  # k=1~4
    for feat_indices in combinations(range(num_features), k):
        # 提取子集
        X_train_sub = X_train[:, feat_indices]
        X_test_sub = X_test[:, feat_indices]

        # 训练线性分类器（这里用LogisticRegression也可以换成LinearSVC）
        clf = LogisticRegression(fit_intercept=True, solver="liblinear")
        clf.fit(X_train_sub, y_train)

        # 提取参数
        weights = clf.coef_[0]  # shape (k,)
        bias = clf.intercept_[0]

        # 输出预测（按 sign(w*x + b)）
        pred_train = np.sign(np.dot(X_train_sub, weights) + bias)
        pred_test = np.sign(np.dot(X_test_sub, weights) + bias)

        acc_train = (pred_train == y_train).mean()
        acc_test = (pred_test == y_test).mean()

        results.append({
            "feature_indices": feat_indices,
            "weights": weights,
            "bias": bias,
            "train_accuracy": acc_train,
            "test_accuracy": acc_test
        })

# === 3. 输出结果表 ===
result_df = pd.DataFrame(results)
result_df["feature_indices"] = result_df["feature_indices"].astype(str)
result_df["weights"] = result_df["weights"].apply(lambda w: np.round(w, 4))
result_df["bias"] = np.round(result_df["bias"], 4)
result_df["train_accuracy"] = np.round(result_df["train_accuracy"], 4)
result_df["test_accuracy"] = np.round(result_df["test_accuracy"], 4)

print(result_df)

# 可保存
result_df.to_csv("feature_combinations_trained_classifiers.csv", index=False)
