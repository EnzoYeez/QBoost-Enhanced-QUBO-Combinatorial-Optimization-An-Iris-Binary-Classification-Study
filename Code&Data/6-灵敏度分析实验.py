# ======================= sensitivity_analysis.py =======================
import numpy as np
import pandas as pd
from collections import Counter
import ast, itertools, pathlib
from statistics import mean, stdev

# === 下面全部是你原来的函数，完全没动 ===
def read_iris_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=["label"]).values
    y = np.where(df["label"].values == 0, -1, 1)
    return X, y

def get_predefined_weak_classifiers():
    return [
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
        {'combo': [0, 1, 2, 3], 'theta': -3.2175}
    ]

def build_replicated_H_matrix(weak_classifiers, X_train_all, y_train, num_repeat=5):
    N = len(y_train)
    all_h_list = []
    for clf in weak_classifiers:
        indices = clf["combo"]
        theta = clf["theta"]
        X_sum = X_train_all[:, indices].sum(axis=1)
        h = np.where(X_sum >= theta, 1, -1)
        for _ in range(num_repeat):
            all_h_list.append(h)
    H_train = np.column_stack(all_h_list)
    return H_train, y_train

def construct_new_QUBO(H, y, lamb=0.1):
    M = H.shape[1]
    Q = np.zeros((M, M))
    c = - H.T @ y + lamb * np.ones(M)
    return Q, c

def simulated_annealing(Q, c, max_iter=10000, T_init=10.0, T_min=1e-3, alpha=0.995, z_init=None, max_vars=11):
    M = Q.shape[0]
    z = z_init.copy() if z_init is not None else np.random.randint(0, 2, size=M)
    energy = z @ Q @ z + c @ z
    T = T_init
    best_z = z.copy()
    best_energy = energy
    for _ in range(max_iter):
        i = np.random.randint(M)
        z_new = z.copy();  z_new[i] = 1 - z_new[i]
        if np.sum(z_new) > max_vars:  # 约束
            continue
        delta = (z_new - z) @ (Q @ z + c)
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            z, energy = z_new, energy + delta
            if energy < best_energy:
                best_energy, best_z = energy, z.copy()
        T = max(T*alpha, T_min)
    return best_z

def build_z_from_ids(used_ids, num_repeat=10, total_classifiers=13):
    M = total_classifiers * num_repeat
    z = np.zeros(M, dtype=int)
    for idx, cnt in Counter(used_ids).items():
        z[idx*num_repeat: idx*num_repeat + cnt] = 1
    return z

def decode_z_to_strong_classifier(z, weak_classifiers, num_repeat=10):
    total_counts = np.zeros(4)        # 4 个特征的累加计数
    thetas        = []               # 被选中弱分类器的 θ 列表
    used_ids      = []               # 记录出现（可能重复）的弱分类器编号

    # --- 遍历 z★ ---
    for i, z_i in enumerate(z):
        if z_i == 1:
            base_idx = i // num_repeat      # 对应弱分类器原始编号
            used_ids.append(base_idx)
            clf = weak_classifiers[base_idx]

            # 累加该弱分类器覆盖的特征
            for f in clf['combo']:
                total_counts[f] += 1

            thetas.append(clf['theta'])

    # --- 如果什么都没选上 ---
    if not thetas:
        return None, None, None, None, None

    # --- 计算强分类器参数 ---
    weights = total_counts / total_counts.sum()
    bias    = -np.mean(thetas)

    # --- 新增统计 ---
    id_counter = Counter(used_ids)      # 每个弱分类器被选次数
    k_unique   = len(id_counter)        # 不同弱分类器种类数

    # 返回值顺序：与你原本保持向后兼容，再附加两个
    return weights, bias, used_ids, id_counter, k_unique

def evaluate_on_test(weights, bias, test_file="iris_test.csv"):
    df = pd.read_csv(test_file)
    X = df.drop(columns=["label"]).values
    y = np.where(df["label"].values == 0, -1, 1)
    acc = (np.sign(X @ weights + bias) == y).mean()
    return acc

def select_random_combo_from_csv(csv_path="weak_to_strong_match_results.csv"):
    raw = pd.read_csv(csv_path).sample(n=1).iloc[0]["best_combo"]
    return list(ast.literal_eval(raw.strip()))

# ======================= 灵敏度分析主流程 =======================
# ======================= 灵敏度分析主流程 =======================
if __name__ == "__main__":
    # 组合列表：可以自行增删
    num_repeat_list = [5, 10, 15]
    lamb_list       = [round(x/10, 1) for x in range(0, 11)]

    # 读取一次数据 & 弱分类器
    X_train, y_train = read_iris_data("iris_train.csv")
    weak_clfs = get_predefined_weak_classifiers()

    results = []
    for num_repeat, lamb in itertools.product(num_repeat_list, lamb_list):
        # 构造 H / Q / c
        H_train, _ = build_replicated_H_matrix(
            weak_clfs, X_train, y_train, num_repeat=num_repeat)
        Q, c = construct_new_QUBO(H_train, y_train, lamb=lamb)

        acc_list, k_total_list, k_unique_list = [], [], []          # <- 新增
        for _ in range(5):  # 每组跑 5 次
            init_ids = select_random_combo_from_csv()
            z0 = build_z_from_ids(
                init_ids, num_repeat=num_repeat,
                total_classifiers=len(weak_clfs))

            z_star = simulated_annealing(Q, c, z_init=z0)

            # 解码返回 5 个变量
            weights, bias, used_ids, id_counter, k_unique = \
                decode_z_to_strong_classifier(
                    z_star, weak_clfs, num_repeat=num_repeat)

            # 准确率
            acc = 0.0 if weights is None else evaluate_on_test(weights, bias)

            # 记录本次试验指标
            acc_list.append(acc)
            k_total_list.append(np.sum(z_star))   # <- 总 1 的个数
            k_unique_list.append(k_unique)        # <- 不同弱分类器种类数

        # 汇总 5 次结果
        line = {
            "num_repeat": num_repeat,
            "lamb": lamb,
            "acc_mean": round(mean(acc_list), 4),
            "acc_std":  round(stdev(acc_list) if len(set(acc_list)) > 1 else 0.0, 4),
            "k_total_mean":   round(mean(k_total_list), 2),   # <- 新增列
            "k_unique_mean":  round(mean(k_unique_list), 2)   # <- 新增列
        }
        results.append(line)

        # 控制台即时打印
        print(f"{num_repeat},{lamb},"
              f"{line['acc_mean']},{line['acc_std']},"
              f"{line['k_total_mean']},{line['k_unique_mean']}")

    # 保存 CSV
    pd.DataFrame(results).to_csv("sensitivity_results.csv", index=False)
    print("\n✅ 已全部完成，结果写入 sensitivity_results1.csv")
