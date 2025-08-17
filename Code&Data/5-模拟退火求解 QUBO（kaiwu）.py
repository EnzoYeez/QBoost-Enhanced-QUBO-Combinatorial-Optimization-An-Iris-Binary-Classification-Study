import numpy as np
import pandas as pd
import kaiwu as kw

def main():
    # === Step 1：初始化 SDK 授权 ===
    kw.license.init(
        user_id="105879997855580162",
        sdk_code="XinpdnsuQLjuPuyon45NHl08PJVEjj"
    )

    # === Step 2：读取 QUBO 矩阵 和 c 向量===
    Q = pd.read_csv("Q.csv", header=None).values
    assert Q.ndim == 2 and Q.shape[0] == Q.shape[1], "Q must be square"
    assert np.allclose(Q, Q.T, atol=1e-6), "Q must be symmetric"

    c = pd.read_csv("c.csv")["bias"].values  # 长度为 M

    # === Step 3：构建变量和 QuboModel ===
    n = Q.shape[0]
    x = kw.qubo.ndarray(n, "x", kw.qubo.Binary)

    # 构建目标函数 E(z) = sum Q_ij x_i x_j
    objective = sum(Q[i][j] * x[i] * x[j] for i in range(n) for j in range(n)) + \
                sum(c[i] * x[i] for i in range(n))  # ✅ 补上 c^T z
    qubo_model = kw.qubo.QuboModel()
    qubo_model.set_objective(objective)


    # === Step 4：配置更强的模拟退火参数 ===
    optimizer = kw.classical.SimulatedAnnealingOptimizer(
        initial_temperature=1e6,
        alpha=0.98,
        cutoff_temperature=1e-4,
        iterations_per_t=1000,
        size_limit=1000,
        rand_seed=42,                    # ✅ 添加随机种子以复现结果
        process_num=-1,                  # ✅ 并行加速（可选）
        flag_evolution_history=False     # ✅ 若需要分析路径则设为 True
    )


    solver = kw.solver.SimpleSolver(optimizer)

    # === Step 5：求解 QUBO ===
    best_z = None
    best_value = float('inf')
    best_solution = None

    for run in range(5):
        solution_dict, qubo_value = solver.solve_qubo(qubo_model)
        # print("solution_dict keys:", list(solution_dict.keys()))
        # z_star = np.array([solution_dict[f"x[{i:02d}]"] for i in range(n)], dtype=int)
        z_star = np.array([solution_dict[f"x[{i:03d}]"] for i in range(n)], dtype=int)


        print(f"[Run {run+1}] z*: {z_star}")
        print(f"  -> E(z): {qubo_value}, selected: {np.sum(z_star)}")

        if qubo_value < best_value and np.sum(z_star) > 0:
            best_value = qubo_value
            best_z = z_star.copy()
            best_solution = solution_dict.copy()

    # Step 6：保存最优解
    if best_z is not None:
        print("\n✅ 最优解（非全零）:")
        print("z* =", best_z)
        print("E(z*) =", best_value)
        print("选中分类器数 =", np.sum(best_z))
        np.savetxt("solution_vector_sanneal.csv", best_z, delimiter=",", fmt="%d")
    else:
        print("❌ 多次退火仍未跳出 z=0，请考虑弱分类器质量或尝试 Tabu 搜索。")


if __name__ == "__main__":
    main()
