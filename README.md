# QBoost-Enhanced QUBO Combinatorial Optimization: An Iris Binary Classification Study

![Project Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Model](https://img.shields.io/badge/model-QBoost%20%2B%20QUBO-orange)
![Dataset](https://img.shields.io/badge/dataset-Iris-9cf)

> **One-liner**: A compact, explainable binary classifier for *Iris (Setosa vs. Versicolor)* built by casting QBoost ensemble selection into a QUBO and solving it via simulated annealing (Kaiwu SDK compatible).

---

## Overview

This project explores a **QBoost** ensemble framed as a **QUBO (Quadratic Unconstrained Binary Optimization)** problem for **binary classification** on the classic Iris dataset (Setosa vs. Versicolor). We construct a library of simple, interpretable **weak classifiers** (thresholded linear projections over 1–4 feature combinations), then **select a sparse subset** by minimizing a QUBO objective that trades training error against model size. The resulting strong classifier is linear in the induced feature space and can be solved on classical or quantum-inspired hardware (e.g., simulated annealing).  
*Key outcomes*: clean modeling pipeline, sparse and interpretable strong classifier, and state-of-the-art performance on this small benchmark. fileciteturn0file0

---

## Highlights

- **End-to-end pipeline**: data cleaning → weak classifier library (130 candidates) → QUBO building → simulated annealing search → evaluation.
- **Sparse & explainable**: explicit \\(\\ell_0\\) / cardinality control promotes compact ensembles.
- **Quantum-friendly**: objective is in standard QUBO form; compatible with simulated/quantum annealers.
- **Strong empirical results**: achieved **100% test accuracy and precision** on Iris (Setosa vs. Versicolor), outperforming SVM, Logistic Regression, Random Forest, XGBoost in this setting. fileciteturn0file0

---

## Installation

```bash
# (recommended) create a fresh environment
conda create -n qboost-qubo-iris python=3.10 -y
conda activate qboost-qubo-iris

# core deps
pip install numpy pandas scikit-learn matplotlib

# (optional) for quantum-inspired / SA backends
# Replace with your vendor SDK or use a local SA implementation.
# Example placeholder:
pip install kaiwu-sdk  # if available (or vendor-provided wheel)
```

---

## Quickstart

```bash
# 1) Prepare the binary subset and standardized splits
python src/build_weak_learners.py --dataset iris --classes setosa versicolor --standardize zscore --out data/

# 2) Build the QUBO (choose λ, K, ρ)
python src/build_qubo.py --lambda 0.85 --K 11 --rho 4.0 --out results/

# 3) Solve with simulated annealing
python src/solve_qubo_sa.py --Q results/Q.npy --c results/c.npy --iters 100000 --alpha 0.995 --t0 10 --tmin 1e-3 --out results/best_subset.json

# 4) Evaluate
python src/evaluate.py --subset results/best_subset.json --train data/iris_train.csv --test data/iris_test.csv --figdir results/figures/
```

> **Tip**: Set a fixed random seed to reduce run-to-run variance when comparing parameter settings.

---

## Results (Iris: Setosa vs. Versicolor)

- **Final model**: sparse strong classifier from QBoost/QUBO selection (\\(K\\approx 11\\)).  
- **Test performance**: **Accuracy 100%**, **Precision 100%** in our runs, outperforming common baselines (SVM, Logistic Regression, Random Forest, XGBoost) on this split. fileciteturn0file0

| Model                | Accuracy | Precision |
|----------------------|---------:|----------:|
| Logistic Regression  |   94.74% |    95.14% |
| SVM                  |   94.74% |    95.14% |
| Random Forest        |   89.47% |    91.11% |
| XGBoost              |   92.11% |    93.05% |
| Neural Network       |   97.37% |    97.44% |
| **QBoost + QUBO (ours)** | **100.00%** | **100.00%** |

> **Sensitivity**: near **λ≈0.85** with **num_repeat=10** (weak-learner snapshots), we observed a strong balance of high accuracy and low variance; overly small λ tended to increase variance. fileciteturn0file0

### Example decision boundaries
We recommend plotting 2D projections (e.g., petal-length vs. petal-width) to visualize linear boundaries induced by the selected set. fileciteturn0file0

---

## Pipeline (Mermaid)

```mermaid
flowchart LR
  A[Data: Iris (Setosa vs. Versicolor)] --> B[Preprocess: clean, z-score, stratified split]
  B --> C[Weak Learners: thresholded linear projections on 1–4 feature combos]
  C --> D[QUBO Build: loss + λ‖z‖₀ + ρ(∑z - K)² → Q, c]
  D --> E[Simulated Annealing: search z*]
  E --> F[Strong Classifier: sign(∑ z*_j h_j(x))]
  F --> G[Evaluate: accuracy, precision, confusion matrix]
```

---

## Configuration

Key hyperparameters:
- `lambda (λ)`: sparsity weight (try 0.6–1.0; sweet spot often around ~0.85). fileciteturn0file0
- `K`: target max number of weak learners (e.g., 11). fileciteturn0file0
- `rho (ρ)`: penalty for exceeding `K` in the soft constraint. fileciteturn0file0
- SA schedule: `t0`, `tmin`, `alpha`, `iters` (cooling & exploration). fileciteturn0file0

---

## References

- Arunachalam & Maity (ICML 2020), *Quantum Boosting*.  
- Neven et al. (ACML 2012), *QBoost: Large Scale Classifier Training with Adiabatic Quantum Optimization*.  
- (Chinese) Report and experiment details from the attached modeling document (APMCM 2025). fileciteturn0file0

---

## License

This work is released under the **MIT License** (see `LICENSE`).

---

## Acknowledgments

- **Kaiwu SDK** for simulated annealing backends and tooling.  
- Classic **Iris** dataset.

---

## Badges & Icons

<p align="center">
  <img alt="QUBO" src="https://img.shields.io/badge/QUBO-✓-informational">
  <img alt="Boosting" src="https://img.shields.io/badge/Boosting-✓-yellow">
  <img alt="Quantum–Inspired" src="https://img.shields.io/badge/Quantum--Inspired-✓-purple">
  <img alt="Explainable" src="https://img.shields.io/badge/Explainable-✓-success">
  <img alt="Reproducible" src="https://img.shields.io/badge/Reproducible-✓-blueviolet">
</p>

<p align="center">
  Made with ❤️  •  Keep it sparse  •  Stay optimal
</p>
