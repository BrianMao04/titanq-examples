# README

#### This project presents examples of using the TitanQ SDK for feature selection for various datasets and compares results to common selection methods
--------------------------------------------------------------------------------


Here is an overview:

- Introduction

- Mathematical Formulation

- TitanQ

- Feature Selection Examples

- License

## Introduction

Feature selection is an important process in machine learning to prevent overfitting, simplify the problem, and improve accuracy, by only selecting those features that are most impactful. Finding the optimal set of features from a feature set is an NP hard optimization problem, motivating InfinityQ to explore an optimization based approach to feature selection. A common way to select is to score each feature independently and select the best ones, implemented in sklearn by SelectKBest. This “filtering” method is only useful for determining importance of individual features. Reformulating as a QUBO takes into account pairwise feature correlation and enables quantum-inspired annealing methods, improving scalability and efficiency. In this way, we can avoid selecting pairs of closely related (redundant) features, giving a better feature set.

## Mathematical Formulation

Much of the mathematical formulation is derived from [A Quantum Feature Selection Method for Network Intrusion Detection](https://ieeexplore.ieee.org/document/9973722) which we will explain below:

Given a dataset, the problem formulation relies on the following parameters:

$\alpha$: The relative weighting between feature importance and independence. At $\alpha = 1$ we only seek to maximize feature importance with no regard to redundancy. The optimal solution in this case would be the same as selecting the $K$ best features according to your scoring function for importance.  

$K$: The number of desired features to select

Now, consider the matrix Q where

```math
Q_{ii} =  - \alpha k  * I(f_{i}, y)
```
```math
Q_{ij} = (1 - \alpha) * I(f_{i}, f_{j})
```

where $I(a, b)$ denotes the correlation between datasets $a$ and $b$. Notice that $I(f_{i}, y)$ is the correlation between feature $i$ and the label (importance) and $I(f_{i}, f_{j})$ measures the correlation between two features (redundancy).

In order to construct our matrix, we calculate $I(a,b)$ using either the Pearson Correlation Coefficient or Mutual Information. Some papers claim Pearson to be the more accurate metric; some claim the same for Mutual Information. It varies by the dataset and the problem.

Then, feature selection is posed as an optimization problem with binary variables where we minimize the following objective function:
```math
E(x) =  x^T Q X = \sum_i - \alpha k  * I(f_{i}, y) + \sum_i \sum_{j \ne i} (1 - \alpha) * I(f_{i}, f_{j})
```
```math
s.t.   \sum_i x_{i} = K
```
where:
```math
x_{i} =
 \begin{cases} 
      1  \text{ if feature $i$ is selected}\\
      0  \text{ otherwise}
   \end{cases}
```

By minimizing the objective function, importance is maximized while redundancy is minimized.

A constraint is added to ensure that exactly K features are selected.

## TitanQ

The hyperparameters used to tune the TitanQ solver are the following:

- *beta* = Scales the problem by this factor (inverse of temperature). A lower *beta* allows for easier escape from local minima, while a higher *beta* is more likely to respect penalties and constraints.

- *coupling_mult* = The strength of the minor embedding for the TitanQ specific hardware.

- *penalty_scaling = The strength of the penalty for violating constraints.

- *timeout_in_secs* = Maximum runtime of the solver in seconds.

- *num_chains* = Number of parallel chains running computations within each engine.

- *num_engines* = Number of independent parallel problems to run. More engines increases the probability of finding an optimal solution.

In addition to the TitanQ hyperparameters, *example.ipynb* also uses *uses_equality* which represents whether the constraint on the number of features selected is formulated as an equality or an inequality. While we do want to ensure that exactly K features are selected, in practice, an inequality constraint may also be used to arrive at a valid selection.

# Feature Selection Examples

*data*: This folder contains the raw training and testing NSL-KDD data downloaded from [Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd), used in *Network_Intrusion.ipynb*.

*weights*: This folder contains the weights matrices used in *Network_Intrusion.ipynb.*

*example.ipynb*: This notebook contains an example for various OpenML datasets.

*dataset_hyperparameters.csv*: Contains pretuned parameters for various datasets for use in *example.ipynb*

*Network_Intrusion.ipynb*: This notebook contains the example on NSL-KDD network intrusion.

*model_generation.py*: This file contains functions to generate weight matrics, preprocess data, and load hyperparameters

The required packages are listed in *requirements.txt* and can be installed using pip:

```bash
pip install -r requirements.txt
```

## License

Released under the Apache License 2.0. See [LICENSE](../LICENSE) file.
