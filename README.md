# lau-statistical-learning

**Statistical learning theory — the mathematical foundations of machine learning.**

A Rust library implementing the core theoretical tools of statistical learning: bias-variance decomposition, VC dimension, PAC learning, Rademacher complexity, cross-validation, regularization (L1/L2/Elastic Net), kernel methods (RBF, polynomial, linear), SVMs (hard & soft margin), learning curves, and multi-armed bandit agent learning. Built on `nalgebra` for linear algebra, 92 tests, full serde support.

---

## What This Does

`lau-statistical-learning` provides **10 modules** covering the mathematical backbone of ML theory:

1. **`bias_variance`** — Decompose prediction error into bias² + variance + noise. Generate theoretical U-shaped tradeoff curves.
2. **`vc_dimension`** — VC dimension descriptors for common hypothesis classes (intervals, linear classifiers, rectangles), growth function bounds (Sauer-Shelah), generalization bounds.
3. **`pac_learning`** — PAC sample complexity for both realizable and agnostic settings, generalization gap verification.
4. **`rademacher`** — Empirical Rademacher complexity via Monte Carlo, Massart's lemma, Rademacher generalization bounds.
5. **`cross_validation`** — K-fold and leave-one-out cross-validation with configurable shuffling and seeds.
6. **`regularization`** — L1 (Lasso), L2 (Ridge), and Elastic Net penalty computation with coefficient paths.
7. **`kernel`** — RBF (Gaussian), polynomial, and linear kernels with a `Kernel` trait, Gram matrix computation.
8. **`svm`** — Hard-margin and soft-margin SVMs with SMO-style optimization, kernelized decision boundaries.
9. **`learning_curves`** — Inverse-root learning curve models, sample complexity estimation, training/test error curves.
10. **`agent_learning`** — Multi-armed bandit (ε-greedy + UCB1), regret analysis, confidence intervals, simulation.

---

## Key Idea

This crate is **not a scikit-learn replacement** — it's the theory layer underneath. Instead of calling `model.fit()`, you compute *why* a model generalizes:

- How many samples do you need? → PAC bounds, VC sample complexity
- How tight is the generalization gap? → Rademacher complexity, VC bounds
- Is your model overfitting or underfitting? → Bias-variance decomposition
- What's the optimal model complexity? → Learning curves, cross-validation
- How does an agent explore vs. exploit? → Multi-armed bandit with regret bounds

Every function returns typed, serializable structs with the relevant parameters — not just a float.

---

## Install

```toml
[dependencies]
lau-statistical-learning = "0.1.0"
```

```sh
cargo add lau-statistical-learning
```

### Requirements

- Rust 2021 edition
- `nalgebra` 0.33 (with `serde-serialize` feature)
- `serde` 1.x + `serde_json` 1.x
- `rand` 0.9 + `rand_distr` 0.5
- `approx` 0.5 (dev-only, for test assertions)

---

## Quick Start

### Bias-Variance Decomposition

```rust
use lau_statistical_learning::*;
use nalgebra::DVector;

let y_true = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
let predictions = vec![
    DVector::from_vec(vec![1.1, 2.1, 2.9, 3.9]),
    DVector::from_vec(vec![0.9, 1.9, 3.1, 4.1]),
    DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
];

let result = bias_variance_decompose(&y_true, &predictions, 0.05);
println!("Bias² = {:.4}", result.bias_sq);
println!("Variance = {:.4}", result.variance);
println!("Noise = {:.4}", result.noise);
println!("Total = {:.4}", result.total_error);
// E[(y - ŷ)²] = Bias² + Var + σ²
```

### VC Dimension & Generalization Bounds

```rust
// Linear classifiers in R² have VC dimension = 3
let vc = VCDimension::new("My classifiers", 3);

// Growth function bound (Sauer-Shelah): m_H(n) ≤ Σ C(n, i)
let growth = vc.growth_function_bound(100);

// Generalization bound with 1000 samples, 95% confidence
let bound = compute_vc_bound(3, 1000, 0.05);
println!("|R(h) - R_emp(h)| ≤ {:.4} with prob ≥ 95%", bound.epsilon);

// Pre-built classes
let intervals = VCDimension::new("Intervals", 2);          // d = 2
let linear_2d = VCClasses::linear_classifiers(2);           // d = 3
let rect_3d = VCClasses::axis_aligned_rectangles(3);        // d = 6
```

### PAC Learning

```rust
// Realizable case: m ≥ (1/ε)(ln|H| + ln(1/δ))
let realizable = pac_bound_sample_size(1000, 0.1, 0.05);
println!("Need {} samples for ε=0.1, δ=0.05", realizable.sample_size);

// Agnostic case: m ≥ C(d + ln(1/δ))/ε²
let agnostic = pac_bound_agnostic(5, 0.1, 0.05);
println!("Agnostic: {} samples with VC dim 5", agnostic.sample_size);

// Verify guarantee
let ok = verify_pac_guarantee(5, 500, 0.2, 0.05);
```

### Rademacher Complexity

```rust
// Empirical Rademacher via Monte Carlo
let predictions = vec![
    vec![1.0, -1.0, 0.5, -0.5],
    vec![-1.0, 1.0, -0.5, 0.5],
    vec![0.0, 0.0, 1.0, -1.0],
];
let rc = rademacher_complexity(&predictions, 1000);

// Massart's lemma: R̂(F) ≤ sqrt(2 ln|F| / n)
let bound = growth_function_bound(100, 50);

// Generalization: R(h) ≤ R̂_emp + 2R̂(F) + 3√(ln(2/δ)/(2n))
let gen = rademacher_generalization_bound(rc, 100, 0.05);
```

### Cross-Validation

```rust
// K-fold
let result = cross_validate_kfold(100, 5, |train, test| {
    // Your scoring function here
    0.95 // placeholder accuracy
}, true);
println!("{}-fold CV: {:.3} ± {:.3}", result.n_folds, result.mean_score, result.std_score);

// Leave-one-out
let loo = cross_validate_loo(50, |train, test| {
    // train has 49 samples, test has 1
    1.0
});
```

### Regularization

```rust
let weights = vec![1.5, -0.3, 2.1, 0.01, -1.8];

let l2 = regularize_l2(&weights, 0.1);   // Ridge: λΣwᵢ²
let l1 = regularize_l1(&weights, 0.1);   // Lasso: λΣ|wᵢ|
let en = regularize_elastic_net(&weights, 0.1, 0.5); // α·L1 + (1-α)·L2

println!("L2 penalty = {:.4}", l2.penalty);
println!("L1 zeroed {} of {} coefficients", l1.zero_count, l1.coefficients.len());
```

### Kernel Methods

```rust
use nalgebra::DVector;

let x = DVector::from_vec(vec![1.0, 2.0]);
let y = DVector::from_vec(vec![3.0, 4.0]);

// RBF (Gaussian) kernel
let rbf = RBFKernel::new(1.0);              // k(x,y) = exp(-γ||x-y||²)
let rbf_s = RBFKernel::with_sigma(2.0);     // γ = 1/(2σ²)
println!("RBF(x,y) = {:.4}", rbf.compute(&x, &y));

// Polynomial kernel
let poly = PolynomialKernel::new(2.0, 1.0);  // (x·y + c)^d
println!("Poly(x,y) = {:.4}", poly.compute(&x, &y));

// Gram matrix
let vecs = vec![x.clone(), y.clone()];
let gram = KernelMatrix::compute(&rbf, &vecs);
// gram is symmetric positive semi-definite n×n matrix
```

### SVM

```rust
let x = DMatrix::from_row_slice(4, 2, &[
    0.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    1.0, 1.0,
]);
let y = DVector::from_vec(vec![-1.0, -1.0, 1.0, 1.0]);

// Hard margin (linearly separable)
let mut svm = HardMarginSVM::new();
svm.fit(&x, &y).unwrap();
let pred = svm.predict(&DVector::from_vec(vec![0.5, 0.5]));

// Soft margin with slack
let mut soft = SoftMarginSVM::new(1.0);  // C = 1.0
soft.fit(&x, &y).unwrap();

// Kernelized SVM
let params = SVMParams::default().with_kernel(Box::new(RBFKernel::new(0.5)));
```

### Learning Curves

```rust
// Generate theoretical learning curve
let curve = learning_curve(100, 0.8, 0.3, 0.05);
for point in &curve.points {
    println!("n={}, train={:.3}, test={:.3}", point.n, point.train_score, point.test_score);
}

// Estimate sample complexity for target performance
let n_needed = sample_complexity_estimate(0.9, 0.8, 0.3, 0.05);
println!("Need ~{} samples for 90% accuracy", n_needed);
```

### Agent Learning (Multi-Armed Bandit)

```rust
let config = AgentLearningConfig {
    n_actions: 3,
    exploration: 0.1,
    n_episodes: 1000,
    ..Default::default()
};

let true_values = vec![0.3, 0.7, 0.5];
let result = simulate_agent_learning(&config, &true_values, 0.05);

println!("Optimal action: {} (true value {:.2})", result.best_action, true_values[result.best_action]);
println!("Avg reward: {:.3}", result.avg_reward);
println!("Regret bound: {:.3}", result.regret_bound);
println!("Action counts: {:?}", result.action_counts);
```

---

## API Reference

### `bias_variance`

| Item | Description |
|---|---|
| `BiasVarianceDecomposition` | Struct: `bias_sq`, `variance`, `noise`, `total_error`, `tradeoff_curve` |
| `TradeoffPoint` | Struct: `complexity`, `bias_sq`, `variance`, `total_error` |
| `bias_variance_decompose(y_true, predictions, noise)` | Core decomposition |
| `generate_tradeoff_curve(n, bias_base, var_base, noise, ...)` | Theoretical U-curve |
| `bias_variance_with_curve(...)` | Decomposition with included tradeoff curve |

### `vc_dimension`

| Item | Description |
|---|---|
| `VCDimension` | Struct: `name`, `d`, `description` |
| `VCBound` | Struct: `d`, `n`, `delta`, `epsilon` |
| `compute_vc_bound(d, n, delta)` | ε = √((8/n)(d·ln(2en/d) + ln(4/δ))) |
| `sample_complexity_vc(d, ε, δ)` | Minimum n for generalization gap ≤ ε |
| `VCClasses` | Factory: `intervals()`, `half_lines()`, `linear_classifiers(d)`, `axis_aligned_rectangles(d)`, `finite_class(log₂|H|)` |

### `pac_learning`

| Item | Description |
|---|---|
| `PACBounds` | Struct: `delta`, `epsilon`, `sample_size`, `vc_dimension` |
| `pac_bound_sample_size(|H|, ε, δ)` | Realizable: m ≥ (1/ε)(ln|H| + ln(1/δ)) |
| `pac_bound_agnostic(d, ε, δ)` | Agnostic: m ≥ C(d + ln(1/δ))/ε² |
| `pac_generalization_gap(d, n, δ)` | Generalization gap bound |
| `verify_pac_guarantee(d, n, ε, δ)` | Boolean check |

### `rademacher`

| Item | Description |
|---|---|
| `rademacher_complexity(predictions, n_trials)` | Monte Carlo empirical Rademacher |
| `rademacher_complexity_estimated(losses, n_trials)` | Rademacher of loss class |
| `growth_function_bound(|F|, n)` | Massart's lemma: √(2 ln|F|/n) |
| `rademacher_generalization_bound(R̂, n, δ)` | R(h) ≤ R̂_emp + 2R̂ + 3√(ln(2/δ)/(2n)) |

### `cross_validation`

| Item | Description |
|---|---|
| `CrossValidationResult` | Struct: `mean_score`, `std_score`, `fold_scores`, `n_folds` |
| `CrossValidator::kfold_indices(n, k, shuffle, seed)` | Generate split indices |
| `cross_validate_kfold(n, k, score_fn, shuffle)` | K-fold CV |
| `cross_validate_loo(n, score_fn)` | Leave-one-out CV |

### `regularization`

| Item | Description |
|---|---|
| `RegularizationResult` | Struct: `penalty`, `coefficients`, `zero_count` |
| `regularize_l1(w, λ)` | Lasso: λΣ|wᵢ| |
| `regularize_l2(w, λ)` | Ridge: λΣwᵢ² |
| `regularize_elastic_net(w, λ, α)` | α·L1 + (1-α)·L2 |

### `kernel`

| Item | Description |
|---|---|
| `Kernel` trait | `compute(x, y) → f64`, `name() → &str` |
| `RBFKernel` | exp(-γ‖x-y‖²), `new(γ)`, `with_sigma(σ)` |
| `PolynomialKernel` | (x·y + c)^d |
| `LinearKernel` | x·y |
| `KernelMatrix::compute(kernel, vectors)` | n×n Gram matrix |
| `rbf_kernel`, `polynomial_kernel`, `linear_kernel` | Convenience free functions |

### `svm`

| Item | Description |
|---|---|
| `SVMParams` | Configuration with kernel, C, tolerance, max iterations |
| `HardMarginSVM` | Max-margin classifier for separable data |
| `SoftMarginSVM` | Slack-variable SVM with penalty C |
| `SVM` trait methods | `fit(X, y)`, `predict(x)`, `decision_function(x)`, `support_vectors()` |

### `learning_curves`

| Item | Description |
|---|---|
| `LearningCurvePoint` | Struct: `n`, `train_score`, `test_score` |
| `learning_curve(n_max, bayes, gap, noise)` | Generate training/test error curves |
| `sample_complexity_estimate(target, bayes, gap, noise)` | Minimum n for target accuracy |

### `agent_learning`

| Item | Description |
|---|---|
| `AgentLearningConfig` | Config: `n_actions`, `exploration` (ε), `n_episodes`, learning rate, discount |
| `AgentLearningModel` | ε-greedy + UCB1 bandit agent |
| `AdaptationResult` | Result: `best_action`, `avg_reward`, `regret_bound`, `action_counts` |
| `simulate_agent_learning(config, true_values, noise)` | Full simulation |
| Methods | `select_action()`, `update(action, reward)`, `compute_regret(optimal)`, `confidence_interval(action, α)`, `finalize(optimal)` |

---

## How It Works

### Bias-Variance Decomposition

Given true values `y` and predictions `ŷ₁, ..., ŷₘ` from different training sets:

```
Mean prediction: ȳ = (1/m) Σₖ ŷₖ
Bias² = (1/n) Σᵢ (ȳᵢ - yᵢ)²
Variance = (1/(m·n)) Σₖ Σᵢ (ŷₖᵢ - ȳᵢ)²
Total = Bias² + Variance + Noise
```

### VC Dimension & Sauer-Shelah Lemma

The growth function `m_H(n)` counts the maximum number of dichotomies on n points. Sauer-Shelah bounds it:

```
m_H(n) ≤ Σᵢ₌₀ᵈ C(n, i)    where d = VC dimension
```

For n ≥ d, this is O(nᵈ). The generalization bound follows:

```
ε = √((8/n)(d·ln(2en/d) + ln(4/δ)))
```

### PAC Learning

**Realizable** (exists h* with zero error): m ≥ (1/ε)(ln|H| + ln(1/δ))
**Agnostic** (no assumption): m ≥ C·(d + ln(1/δ))/ε²

The agnostic bound scales as 1/ε² (quadratically worse) because we can't assume the target is in H.

### Rademacher Complexity

The empirical Rademacher complexity measures the richness of a function class by correlating it with random ±1 noise:

```
R̂(F) = E_σ [sup_{f∈F} (1/n) Σ σᵢ f(xᵢ)]
```

Estimated via Monte Carlo: generate random σ vectors, compute the sup for each, average. Massart's lemma gives a closed-form upper bound: √(2 ln|F|/n).

### Cross-Validation

K-fold splits n samples into k roughly equal partitions. Each partition serves as the test set once, with the remaining k-1 folds as training. The mean and standard deviation of fold scores estimate generalization performance. LOO-CV is the special case k = n.

### Regularization

- **L2 (Ridge):** Penalty = λ·Σwᵢ². Shrinks all weights proportionally. Differentiable → easy optimization.
- **L1 (Lasso):** Penalty = λ·Σ|wᵢ|. Produces sparse solutions (drives small weights to exactly 0).
- **Elastic Net:** α·L1 + (1-α)·L2. Combines sparsity (L1) with grouping stability (L2).

### Kernel Methods

Kernels implicitly map data into high-dimensional feature spaces without computing the mapping explicitly. The kernel trick: any algorithm that depends only on dot products can be "kernelized" by replacing ⟨x, y⟩ with k(x, y).

- **RBF:** k(x,y) = exp(-γ‖x-y‖²) → infinite-dimensional feature space
- **Polynomial:** k(x,y) = (⟨x,y⟩ + c)^d → all polynomial features up to degree d
- **Linear:** k(x,y) = ⟨x,y⟩ → no feature mapping

### SVM

Hard-margin SVM solves: minimize ½‖w‖² subject to yᵢ(w·xᵢ + b) ≥ 1 for all i. Soft-margin adds slack variables ξᵢ with penalty C. The dual form only requires kernel evaluations, enabling nonlinear boundaries via the kernel trick.

### Agent Learning (Multi-Armed Bandit)

ε-greedy: with probability ε, explore a random action; otherwise exploit the current best. UCB1: select action maximizing Q(a) + √(2 ln t / nₐ), which balances exploitation and exploration with a theoretically optimal regret bound of O(√(T·K·ln T)).

---

## The Math

### Bias-Variance Decomposition

```
E[(y - ŷ)²] = Bias[ŷ]² + Var[ŷ] + σ²

where:
  Bias²[ŷ] = E[(ŷ - y)²]  (systematic error)
  Var[ŷ] = E[(ŷ - E[ŷ])²]  (sensitivity to training set)
  σ² = irreducible noise
```

### VC Generalization Bound

```
With probability ≥ 1 - δ:
  |R(h) - R_emp(h)| ≤ √((8/n)(d·ln(2en/d) + ln(4/δ)))

where d = VC dimension, n = sample size
```

### PAC Sample Complexity

```
Realizable: m(ε, δ) = ⌈(1/ε)(ln|H| + ln(1/δ))⌉
Agnostic:   m(ε, δ) = ⌈(C/ε²)(d + ln(1/δ))⌉     (C ≈ 8)
```

### Rademacher Generalization

```
R(h) ≤ R̂_emp(h) + 2·R̂(F) + 3·√(ln(2/δ) / (2n))

Massart's lemma: R̂(F) ≤ √(2·ln|F| / n)
```

### RBF Kernel

```
k(x, y) = exp(-γ·‖x - y‖²)

With bandwidth σ: γ = 1/(2σ²)
```

### L1/L2/Elastic Net Penalties

```
L2: Ω(w) = λ·Σ wᵢ²
L1: Ω(w) = λ·Σ |wᵢ|
EN: Ω(w) = λ·(α·Σ|wᵢ| + (1-α)·Σwᵢ²)
```

### SVM Primal (Soft Margin)

```
min  ½‖w‖² + C·Σ ξᵢ
s.t. yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ,  ξᵢ ≥ 0
```

### UCB1 Regret Bound

```
E[Regret_T] ≤ Σᵢ:Δᵢ>0 (8·ln T / Δᵢ) + (1 + π²/3)·Σ Δᵢ

where Δᵢ = μ* - μᵢ (gap from optimal)
```

---

## License

MIT
