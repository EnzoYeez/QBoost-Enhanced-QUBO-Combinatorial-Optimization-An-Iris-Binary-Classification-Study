import numpy as np
import pandas as pd

# === Step 1: 读取训练集数据 ===
def read_iris_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=["label"]).values  # shape: (N, d)
    y = np.where(df["label"].values == 0, -1, 1)  # 标签变成 ±1
    return X, y

# === Step 2: 构造 130 个弱分类器结构矩阵 A 和阈值向量 theta ===
def get_structured_classifiers(num_repeat=10):
    base_classifiers = [
        {'combo': [0], 'theta': -1.3701},
        {'combo': [2], 'theta': -1.1090},
        {'combo': [3], 'theta': 0.8868},
        {'combo': [0, 2], 'theta': -2.4102},
        {'combo': [0, 3], 'theta': -2.4403},
        {'combo': [1, 2], 'theta': -0.2110},
        {'combo': [1, 3], 'theta': -0.8105},
        {'combo': [2, 3], 'theta': -2.1825},
        {'combo': [0, 1, 2], 'theta': -2.0623},
        {'combo': [0, 1, 3], 'theta': -2.1494},
        {'combo': [0, 2, 3], 'theta': -3.4112},
        {'combo': [1, 2, 3], 'theta': -2.1510},
        {'combo': [0, 1, 2, 3], 'theta': -3.2175},
    ]
    d = 4
    M = len(base_classifiers) * num_repeat
    A = np.zeros((M, d))
    theta = np.zeros(M)

    for i, base in enumerate(base_classifiers):
        for r in range(num_repeat):
            idx = i * num_repeat + r
            A[idx, base['combo']] = 1
            theta[idx] = base['theta']  # 可选扰动项 + np.random.normal(0, ε)
    return A, theta

# === Step 3: 构造 QUBO 矩阵 ===
def construct_QUBO_matrix(X, y, A, theta, lamb=0.2):
    """
    对频率重构形式 f(x; z) = w(z)^T x + b(z)
    使用二次近似拟合 margin-based loss，构造 Q 和 c。
    """
    M, d = A.shape
    N = X.shape[0]

    # Step 1: 计算频率向量投影结果
    XA = X @ A.T      # shape: (N, M)
    yt = y.reshape(-1, 1)  # (N,1)

    # Step 2: 构造 Q: 协方差项 sum_i <x_i^T a_j, x_i^T a_k>
    Q = XA.T @ XA  # shape: (M, M)

    # Step 3: 构造 c: 标签相关性项 -2 y^T X a_j + λ
    c = -2 * (yt.T @ XA).flatten() + lamb

    return Q, c

# === Step 4: 主函数 ===
if __name__ == "__main__":
    # 1. 加载数据
    X_train, y_train = read_iris_data("iris_train.csv")

    # 2. 构造结构矩阵 A 和 θ 向量
    A, theta = get_structured_classifiers(num_repeat=10)
    print("✅ 构造完成 A, shape =", A.shape)  # (130, 4)

    # 3. 构造 QUBO 矩阵 Q 和向量 c
    Q, c = construct_QUBO_matrix(X_train, y_train, A, theta, lamb=0.2)
    print("✅ QUBO 构造完成: Q shape =", Q.shape, ", c shape =", c.shape)

    # 4. 保存输出
    pd.DataFrame(Q).to_csv("Q.csv", index=False, header=False)
    pd.DataFrame(c).to_csv("c.csv", index=False, header=False)
    pd.DataFrame(A).to_csv("A.csv", index=False, header=False)
    pd.DataFrame(theta).to_csv("theta.csv", index=False, header=False)
    print("✅ 所有 QUBO 参数文件已保存：Q.csv, c.csv, A.csv, theta.csv")
