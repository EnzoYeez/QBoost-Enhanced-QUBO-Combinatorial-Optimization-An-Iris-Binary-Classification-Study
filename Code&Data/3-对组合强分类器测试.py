import pandas as pd
import numpy as np

# === 读取匹配结果和测试数据 ===
match_df = pd.read_csv("weak_to_strong_match_results.csv")
test_df = pd.read_csv("iris_test.csv")

# === 提取测试集特征与标签 ===
X_test = test_df.drop(columns=["label"]).values
y_test = np.where(test_df["label"].values == 0, -1, 1)  # 映射为 -1 / +1

results = []

print(f"📊 测试集中样本数量: {len(y_test)}")
print(f"🔍 开始对组合构造的强分类器进行测试...\n")

for _, row in match_df.iterrows():
    strong_id = row["strong_id"]
    matched = row["matched"]

    # 加载重构权重和偏置
    weights = np.array(eval(row["reconstructed_weights"]))  # 例如: [0.33, 0.22, 0.22, 0.22]
    bias = row["reconstructed_bias"]

    # 线性组合并预测
    logits = np.dot(X_test, weights) + bias
    y_pred = np.where(logits >= 0, 1, -1)

    # 准确率计算
    acc = (y_pred == y_test).mean()

    results.append({
        "strong_id": strong_id,
        "matched": True,
        "accuracy": round(acc, 4)
    })
    print(f"✅ 强分类器 #{strong_id}: 测试准确率 = {acc:.4f}")

# === 保存测试结果 ===
df_acc = pd.DataFrame(results)
df_acc.to_csv("weak_combo_test_accuracy.csv", index=False)
print("\n📁 所有准确率结果已保存到：weak_combo_test_accuracy.csv")
