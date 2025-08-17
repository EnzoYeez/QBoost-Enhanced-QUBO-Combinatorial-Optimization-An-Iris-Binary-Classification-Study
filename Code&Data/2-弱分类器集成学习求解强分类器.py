import itertools
import numpy as np
import pandas as pd

# ==== 弱分类器列表 ====
weak_classifiers = [
    {'features': (0,), 'theta': -1.3701},
    {'features': (2,), 'theta': -1.1090},
    {'features': (3,), 'theta': 0.8868},
    {'features': (0, 2), 'theta': -2.4102},
    {'features': (0, 3), 'theta': -2.4403},
    {'features': (1, 2), 'theta': -0.2110},
    {'features': (1, 3), 'theta': -0.8105},
    {'features': (2, 3), 'theta': -2.1825},
    {'features': (0, 1, 2), 'theta': -2.0623},
    {'features': (0, 1, 3), 'theta': -2.1494},
    {'features': (0, 2, 3), 'theta': -3.4112},
    {'features': (1, 2, 3), 'theta': -2.1510},
    {'features': (0, 1, 2, 3), 'theta': -3.2175},
]

# ==== 强分类器表 ====
strong_classifiers = [
    {"features": (0,), "weights": [2.5561], "bias": 0.2667},
    {"features": (1,), "weights": [-2.5651], "bias": 0.1065},
    {"features": (2,), "weights": [3.2294], "bias": 0.2986},
    {"features": (3,), "weights": [3.2563], "bias": 0.2613},
    {"features": (0,1), "weights": [2.3379, -2.2091], "bias": 0.3299},
    {"features": (0,2), "weights": [0.7241, 2.9455], "bias": 0.4107},
    {"features": (0,3), "weights": [0.8859, 2.9125], "bias": 0.3549},
    {"features": (1,2), "weights": [-1.2923, 2.6508], "bias": 0.1327},
    {"features": (1,3), "weights": [-1.3774, 2.6052], "bias": 0.1081},
    {"features": (2,3), "weights": [1.9652, 1.9138], "bias": 0.3224},
    {"features": (0,1,2), "weights": [1.0905, -1.4318, 2.0367], "bias": 0.2526},
    {"features": (0,1,3), "weights": [1.1195, -1.451, 1.9949], "bias": 0.1961},
    {"features": (0,2,3), "weights": [0.5027, 1.8413, 1.8349], "bias": 0.3925},
    {"features": (1,2,3), "weights": [-1.0975, 1.6183, 1.6264], "bias": 0.1471},
    {"features": (0,1,2,3), "weights": [0.7526, -1.1805, 1.3731, 1.4231], "bias": 0.2144},
]

# ==== 参数 ====
max_weak_clf = 11
n_features = 4
threshold = 0.3

print(f"🚀 开始匹配共 {len(strong_classifiers)} 个强分类器...")

records = []

for i, strong in enumerate(strong_classifiers):
    print(f"\n🔎 正在处理强分类器 #{i+1}/{len(strong_classifiers)}: 特征 {strong['features']} ...")
    best_combo = None
    best_weights = None
    best_bias = None
    min_error = float("inf")

    combos = []
    for k in range(1, max_weak_clf + 1):
        combos.extend(itertools.combinations_with_replacement(range(len(weak_classifiers)), k))
    print(f"   ⏳ 共生成组合数量：{len(combos)}，开始搜索...")

    for idx, combo in enumerate(combos):
        if idx % 1000 == 0:
            print(f"     ... 已处理 {idx}/{len(combos)} 组组合")

        feature_count = np.zeros(n_features)
        thetas = []
        for clf_idx in combo:
            clf = weak_classifiers[clf_idx]
            for f in clf["features"]:
                feature_count[f] += 1
            thetas.append(clf["theta"])
        
        total = feature_count.sum()
        if total == 0:
            continue
        weights = feature_count / total
        bias = -np.mean(thetas)

        target_idx = np.array(strong["features"])
        weights_sub = weights[target_idx]
        target_weights = np.array(strong["weights"])
        error = np.linalg.norm(weights_sub - target_weights / np.linalg.norm(target_weights))
        bias_error = abs(bias - strong["bias"])
        total_error = error + bias_error

        if total_error < min_error:
            min_error = total_error
            best_combo = combo
            best_weights = weights
            best_bias = bias

    # ==== 打印并保存 ====
    matched = min_error < threshold

    print(f"\n✅ 匹配完成，最佳误差 = {min_error:.4f}")
    print(f"   {'✅' if matched else '❌'} 最佳组合: {best_combo}")
    print(f"   → 重构权重: {np.round(best_weights, 4)}")
    print(f"   → 重构 bias: {round(best_bias, 4)}")

    records.append({
        "strong_id": i + 1,
        "features": str(strong["features"]),
        "true_weights": np.round(strong["weights"], 4).tolist(),
        "true_bias": round(strong["bias"], 4),
        "matched": matched,
        "best_combo": str(best_combo),
        "reconstructed_weights": np.round(best_weights, 4).tolist(),
        "reconstructed_bias": round(best_bias, 4),
        "total_error": round(min_error, 6)
    })

# ==== 保存为 CSV ====
df_results = pd.DataFrame(records)
df_results.to_csv("weak_to_strong_match_results.csv", index=False)
print("\n📄 所有强分类器的最佳匹配结果已保存至：weak_to_strong_match_results.csv")
