import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# === 步骤 1：读取数据 ===
df = pd.read_csv("iris.data", header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']

# === 步骤 2：保留两类样本，并映射为 0 和 1 ===
df = df[df['label'].isin(['Iris-setosa', 'Iris-versicolor'])]
df['label'] = df['label'].map({'Iris-setosa': 0, 'Iris-versicolor': 1})

print(f"原始样本数（仅2类）：{len(df)}")

# === 步骤 3：检查并删除缺失值（保留前打印） ===
missing_info = df[df.isnull().any(axis=1)]
if not missing_info.empty:
    print(f"\n❗共发现 {len(missing_info)} 条缺失值样本，已删除，具体如下：")
    print(missing_info)
    df = df.dropna()
else:
    print("\n✅ 没有缺失值")

# === 步骤 4：检查并删除重复（保留第一条）===
duplicate_all = df[df.duplicated(keep=False)]
if not duplicate_all.empty:
    print(f"\n❗共发现 {len(duplicate_all)} 条重复记录（含原始+副本），具体如下：")
    print(duplicate_all)

    duplicate_later = df[df.duplicated(keep='first')]
    print(f"\n❗准备删除 {len(duplicate_later)} 条多余副本，具体如下：")
    print(duplicate_later)

    df = df.drop_duplicates(keep='first')
else:
    print("\n✅ 没有重复样本")

print(f"\n清洗后样本数：{len(df)}")


# === 步骤 5：基于 IQR 进行离群点检测与剔除（标准化前） ===
def remove_outliers_iqr(df, feature_cols):
    outlier_indices = set()
    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        # 找出该特征中的离群值索引
        outliers = df[(df[col] < lower) | (df[col] > upper)].index
        outlier_indices.update(outliers)
        print(f"特征 {col} 的离群值范围：({lower:.2f}, {upper:.2f})，发现 {len(outliers)} 个离群样本")

    print(f"\n❗总共检测出 {len(outlier_indices)} 个样本存在离群值，已剔除")
    return df.drop(index=outlier_indices)

# 调用离群点剔除
feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df = remove_outliers_iqr(df, feature_columns)
print(f"剔除离群值后样本数：{len(df)}")

# === 步骤 6：标准化特征 ===
X = df.drop(columns='label')
y = df['label']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df['label'] = y.values

# === 步骤 6：划分训练集和测试集并保存为 CSV 文件 ===
train_df, test_df = train_test_split(
    processed_df, test_size=0.2, random_state=42, stratify=processed_df['label']
)
processed_df.to_csv("iris_binary_processed.csv", index=False)
train_df.to_csv("iris_train.csv", index=False)
test_df.to_csv("iris_test.csv", index=False)

# === 步骤 8：打印结果确认 ===
print("\n✅ 所有数据文件已保存：")
print("- iris_binary_processed.csv（标准化后完整数据）")
print("- iris_train.csv（训练集）")
print("- iris_test.csv（测试集）")
