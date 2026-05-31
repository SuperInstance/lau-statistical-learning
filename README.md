# lau-statistical-learning

Statistical learning theory — mathematical foundations of machine learning.

## Features

- **Bias-variance tradeoff**: Decomposition and theoretical curve generation
- **VC dimension**: Generalization bounds for common hypothesis classes
- **PAC learning**: Realizable and agnostic sample complexity bounds
- **Rademacher complexity**: Empirical estimation and generalization bounds
- **Cross-validation**: K-fold and leave-one-out with configurable shuffling
- **Regularization**: L1 (Lasso), L2 (Ridge), and elastic net via proximal operators
- **Kernel methods**: RBF, polynomial, linear kernels with kernel matrix computation
- **Support vector machines**: Hard and soft margin SVM via SMO optimization
- **Learning curves**: Theoretical curves, sample complexity estimation, overfitting detection
- **Agent learning**: Multi-armed bandit analysis, UCB1, regret bounds, exploration-exploitation

## Usage

```rust
use lau_statistical_learning::*;

// Bias-variance decomposition
let y_true = DVector::from_vec(vec![1.0, 2.0, 3.0]);
let predictions = vec![
    DVector::from_vec(vec![1.1, 2.1, 2.9]),
    DVector::from_vec(vec![0.9, 1.9, 3.1]),
];
let bv = bias_variance_decompose(&y_true, &predictions, 0.05);

// K-fold cross-validation
let result = cross_validate_kfold(100, 5, |train, test| {
    // Your scoring function here
    0.95
}, true);

// SVM training
let svm = SoftMarginSVM::train(&features, &labels, SVMParams::default());
let prediction = svm.predict(&new_point);
```

## Dependencies

- `nalgebra` — Linear algebra
- `serde` — Serialization
- `rand` — Random number generation

## License

MIT
