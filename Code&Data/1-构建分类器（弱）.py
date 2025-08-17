import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# === 加载数据 ===
train_df = pd.read_csv("iris_train.csv")
test_df = pd.read_csv("iris_test.csv")

X_train_all = train_df.drop(columns=["label"]).values
y_train = np.where(train_df["label"].values == 0, -1, 1)
X_test_all = test_df.drop(columns=["label"]).values
y_test = np.where(test_df["label"].values == 0, -1, 1)

# === 构建所有特征组合（1~4维） ===
feature_names = train_df.columns[:-1]
feature_indices = list(range(len(feature_names)))

all_combinations = []
for r in range(1, len(feature_indices) + 1):
    all_combinations.extend(itertools.combinations(feature_indices, r))

print(f"\n共生成特征组合 {len(all_combinations)} 个（1~4维）\n")

# === 存储弱分类器信息 ===
weak_classifier_records = []

# === 遍历组合并训练 ===
for idx, combo in enumerate(all_combinations):
    feature_list = [feature_names[i] for i in combo]
    X_sum_train = X_train_all[:, combo].sum(axis=1)
    X_sum_test = X_test_all[:, combo].sum(axis=1)

    theta_candidates = np.linspace(X_sum_train.min(), X_sum_train.max(), 1000)
    found = False

    for theta in theta_candidates:
        pred_train = np.where(X_sum_train >= theta, 1, -1)
        acc_train = accuracy_score(y_train, pred_train)

        if 0.60 <= acc_train <= 0.80:
            pred_test = np.where(X_sum_test >= theta, 1, -1)
            acc_test = accuracy_score(y_test, pred_test)

            print(f"✅ 弱分类器 #{idx+1}（特征组合: {feature_list}）")
            print(f" - 最佳阈值 θ: {theta:.4f}")
            print(f" - 训练集准确率: {acc_train:.3f}")
            print(f" - 测试集准确率:  {acc_test:.3f}\n")

            # 添加记录
            weak_classifier_records.append({
                "id": idx + 1,
                "feature_indices": list(combo),
                "feature_names": '+'.join(feature_list),
                "theta": round(theta, 4),
                "acc_train": round(acc_train, 3),
                "acc_test": round(acc_test, 3)
            })

            found = True
            break

    if not found:
        print(f"❌ 特征组合 {feature_list} 未能成为弱分类器（无 θ 满足条件）\n")

# === 保存为 CSV 文件 ===
df_weak = pd.DataFrame(weak_classifier_records)
df_weak.to_csv("weak_classifiers_info.csv", index=False)
print(f"\n已保存 {len(df_weak)} 个弱分类器信息至 weak_classifiers_info.csv")
