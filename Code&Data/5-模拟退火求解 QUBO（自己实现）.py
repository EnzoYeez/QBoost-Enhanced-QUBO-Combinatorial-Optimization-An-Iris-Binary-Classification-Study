import numpy as np  
import pandas as pd
from collections import Counter
import ast

# === 模拟退火求解器（布尔向量 + 频率展开）===
def simulated_annealing(Q, c, max_iter=10000, T_init=10.0, T_min=1e-3, alpha=0.995, z_init=None, max_vars=11):
    M = Q.shape[0]
    z = z_init.copy() if z_init is not None else np.random.randint(0, 2, size=M)
    energy = z @ Q @ z + c @ z
    best_z = z.copy()
    best_energy = energy
    T = T_init

    for step in range(max_iter):
        i = np.random.randint(M)
        z_new = z.copy()
        z_new[i] = 1 - z_new[i]

        if np.sum(z_new) > max_vars:
            continue  # 限制最大变量数量

        delta = (z_new - z) @ (Q @ z + c)
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            z = z_new
            energy += delta
            if energy < best_energy:
                best_energy = energy
                best_z = z.copy()
        T = max(T * alpha, T_min)

    return best_z, best_energy

# === 弱分类器定义（原始13个）===
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

# === 初始化向量 z（长度 = M × repeat）===
def build_z_from_ids(used_ids, num_repeat=10, total_classifiers=13):
    M = total_classifiers * num_repeat
    z = np.zeros(M, dtype=int)
    counter = Counter(used_ids)
    for idx, count in counter.items():
        start = idx * num_repeat
        z[start:start + count] = 1
    return z

# === 解码 z（频率重构的强分类器）===
def decode_z_to_strong_classifier(z, weak_classifiers, num_repeat=10):
    total_counts = np.zeros(4)  # 特征维度为 4
    selected_thetas = []
    used_ids = []

    for i, z_i in enumerate(z):
        if z_i == 1:
            base_idx = i // num_repeat
            used_ids.append(base_idx)
            clf = weak_classifiers[base_idx]
            for f in clf['combo']:
                total_counts[f] += 1
            selected_thetas.append(clf['theta'])

    total_votes = np.sum(total_counts)
    if total_votes == 0:
        print("❌ 没有选中任何弱分类器")
        return None, None, None, None

    weights = total_counts / total_votes
    bias = -np.mean(selected_thetas)
    id_counter = Counter(used_ids)
    return weights, bias, used_ids, id_counter

# === 评估测试集准确率 ===
def evaluate_on_test(weights, bias, test_file="iris_test.csv"):
    df = pd.read_csv(test_file)
    X = df.drop(columns=["label"]).values
    y = np.where(df["label"].values == 0, -1, 1)
    pred = np.sign(X @ weights + bias)
    acc = (pred == y).mean()
    return acc

# === 从CSV中解析 best_combo 初始组合 ===
def select_random_combo_from_csv(csv_path="weak_to_strong_match_results.csv"):
    df = pd.read_csv(csv_path)
    row = df.sample(n=1).iloc[0]
    combo_str = row["best_combo"]
    try:
        combo = list(ast.literal_eval(combo_str.strip()))
        if not all(isinstance(x, int) for x in combo):
            raise ValueError("非整数组合")
        return combo
    except Exception as e:
        print(f"⚠️ 解析失败: {combo_str}")
        raise e

# === 主程序入口 ===
if __name__ == "__main__":
    # 参数设置
    num_repeat = 10
    max_vars = 11

    # 1. 读取数据
    Q = pd.read_csv("Q.csv", header=None).values
    c = pd.read_csv("c.csv", header=None).values.flatten()
    weak_classifiers = get_predefined_weak_classifiers()

    # 2. 初始化 z
    init_combo = select_random_combo_from_csv()
    print("🎯 初始组合:", init_combo)

    z_init = build_z_from_ids(init_combo, num_repeat=num_repeat, total_classifiers=len(weak_classifiers))
    print("✅ 构造 z_init 成功，非零数 =", np.sum(z_init))

    # 3. 模拟退火求解
    z_star, energy = simulated_annealing(Q, c, z_init=z_init, max_vars=max_vars)
    print("✅ 最优解 energy =", round(energy, 4))
    print("选中变量数 =", int(np.sum(z_star)))

    # 4. 解码为强分类器
    weights, bias, used_ids, counter = decode_z_to_strong_classifier(z_star, weak_classifiers, num_repeat)
    if weights is not None:
        print("\n✅ 强分类器构造成功：")
        print("➡️ 选中弱分类器（含重复）:", used_ids)
        print("➡️ 各弱分类器频次：")
        for idx, count in counter.items():
            combo = weak_classifiers[idx]["combo"]
            theta = weak_classifiers[idx]["theta"]
            print(f"  - #{idx} 特征={combo} θ={theta} 次数={count}")
        print("➡️ 最终权重:", np.round(weights, 4))
        print("➡️ 最终 bias:", round(bias, 4))

        acc = evaluate_on_test(weights, bias)
        print("✅ 测试集准确率 =", round(acc, 4))
