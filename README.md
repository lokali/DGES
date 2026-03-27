# [NeurIPS 2024] On Causal Discovery in the Presence of Deterministic Relations

This is the implementation of synthetic experiments for the paper **"[On Causal Discovery in the Presence of Deterministic Relations](https://openreview.net/forum?id=pfvcsgFrJ6)"**, NeurIPS 2024.


If you find it useful, please consider citing:

```
@inproceedings{li2024causal,
  title={On Causal Discovery in the Presence of Deterministic Relations},
  author={Li, Loka and Dai, Haoyue and Al Ghothani, Hanin and Huang, Biwei and Zhang, Jiji and Harel, Shahar and Bentwich, Isaac and Chen, Guangyi and Zhang, Kun},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
  year={2024}
}
```

## Overview

- We find that **score-based methods with exact search** can naturally address the issues of deterministic relations, given that the SMR assumption is met.

- To improve efficiency, we propose the novel and **versatile framework** called Determinism-aware Greedy Equivalent Search (DGES),
    - encompassing both linear and nonlinear models,
    - continuous and discrete data types,
    - Gaussian and non-Gaussian data distributions.

- We provide the **identifiability conditions** of DGES under general functional models.


## DGES is now integrated into causal-learn

DGES has been integrated into the [causal-learn](https://github.com/py-why/causal-learn) package. You can now use DGES with a single function call:

```python
from causallearn.search.ScoreBased.DGES import dges

# Run DGES on your data (Phase 1 + 2, fast)
Record = dges(X, score_func='local_score_BIC_from_cov_deterministic',
              det_threshold=1e-5, det_epsilon=0.01)

# The learned causal graph (CPDAG)
G = Record['G']

# Detected minimal deterministic clusters (MinDCs)
mindcs = Record['mindcs']
```

### Installation

```sh
pip install causal-learn
```

### Usage

#### Basic Usage (Phase 1 + 2, fast)

```python
import numpy as np
from causallearn.search.ScoreBased.DGES import dges

# X: data matrix with shape (n_samples, n_features)
Record = dges(X)

# Access results
G = Record['G']                    # learned CPDAG (GeneralGraph)
mindcs = Record['mindcs']          # detected MinDCs
score = Record['score']            # final score
updates_fwd = Record['update1']    # forward step updates
updates_bwd = Record['update2']    # backward step updates
```

#### Full 3-Phase Algorithm (with Exact Search)

For better accuracy on small graphs, enable Phase 3 (exact search on DC neighborhoods):

```python
Record = dges(X,
              score_func='local_score_BIC_from_cov_deterministic',
              det_threshold=1e-5,    # threshold for detecting deterministic relations
              det_epsilon=0.01,      # epsilon for modified BIC: log(sigma + epsilon)
              skip_exact_search=False,
              exact_search_method='astar')

# Additional outputs when exact search is enabled
exact_nodes = Record['exact_search_nodes']  # nodes in exact search subgraph
```

#### With Nonlinear Score Functions

DGES also supports nonlinear score functions for data with nonlinear causal relationships:

```python
Record = dges(X,
              score_func='local_score_CV_general',
              det_threshold=0.999)   # R^2 threshold for nonlinear DC detection
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `X` | (required) | Data matrix, shape `(n_samples, n_features)` |
| `score_func` | `'local_score_BIC_from_cov_deterministic'` | Score function name |
| `maxP` | `None` | Maximum number of parents |
| `parameters` | `None` | Score function parameters (e.g., `{'lambda_value': 0.5}`) |
| `node_names` | `None` | Names for variables |
| `det_threshold` | `1e-5` | Threshold for deterministic relation detection |
| `det_epsilon` | `0.01` | Epsilon for modified BIC: `log(sigma + epsilon)` |
| `exact_search_method` | `'astar'` | `'astar'` or `'dp'` for Phase 3 |
| `skip_exact_search` | `True` | Set `False` to enable Phase 3 (exact search) |

### Returns

A dictionary with:

| Key | Description |
|---|---|
| `'G'` | Learned causal graph (CPDAG, `GeneralGraph`) |
| `'update1'` | Forward step updates (insert operations) |
| `'update2'` | Backward step updates (delete operations) |
| `'G_step1'` | Graph at each forward step |
| `'G_step2'` | Graph at each backward step |
| `'score'` | Final score of the learned graph |
| `'mindcs'` | Detected minimal deterministic clusters |
| `'det_clusters'` | Deterministic cluster assignments |
| `'exact_search_nodes'` | Nodes in exact search subgraph (if Phase 3 enabled) |

### Score Functions

| Score Function | Data Type | Description |
|---|---|---|
| `'local_score_BIC_from_cov_deterministic'` | Linear, continuous | Modified BIC with epsilon (recommended) |
| `'local_score_BIC_from_cov'` | Linear, continuous | Standard BIC (without DC handling) |
| `'local_score_CV_general'` | Nonlinear, continuous | Cross-validated kernel score |
| `'local_score_marginal_general'` | Nonlinear, continuous | Marginal likelihood kernel score |
| `'local_score_BDeu'` | Discrete | BDeu score |


## DGES Algorithm

- **DGES in theory** comprises three phases:
    - (1) identify minimal deterministic clusters (i.e., the minimal set of variables with deterministic relationships);
    - (2) run modified Greedy Equivalent Search (GES) to obtain an initial graph, and
    - (3) perform exact search exclusively on each deterministic cluster and its neighbors.

- **Modified GES** mainly has two changes:

    - Modified BIC score functions:
        - Before: $\log L \propto -\frac{n}{2}(1+ \log |\Sigma|)$
        - After: $\log L \propto -\frac{n}{2}(1+ \log |\Sigma+\epsilon|)$;

    - Modified edge adding and deleting:
        - **Forward**: when parent set of $X_j$ determines $X_i$ (i.e., $X_i$ is a deterministic function of $\text{PA}(X_j)$), force add edge $X_i \to X_j$.
        - **Backward**: when $X_i$ and $X_j$ are in the same MinDC, protect the edge from deletion.

- **Efficient DGES in Implementation**: As you may notice, the first phase could be time-consuming, therefore, we update our code (in the current github version) by: exchanging the first and second phases, so that the search space is smaller for detecting MinDCs.

- **Another Efficient Implementation**: If you are running a large-scale graph, then you may just use GES + modified BIC score, to quickly get an approximate solution. You can skip the MinDC-detection phase and exact-search phase, only running the modified-GES phase without modified-edge-adding-and-deleting step.


## Reproducing Paper Experiments

The original experiment scripts are also included for reproducing results from the paper:

```sh
# Set up environment
conda create -n DGES python=3.8
conda activate DGES
pip install causal-learn numpy pydot networkx igraph torch

# Run experiments with the original csl-based code
python run_dges_linear_1DC.py    # Linear model with one MinDC
python run_dges_linear_2DC.py    # Linear model with two MinDCs
python run_dges_nonlinear_1DC.py # Nonlinear model with one MinDC
python run_dges_nonlinear_2DC.py # Nonlinear model with two MinDCs

# Run experiments with causal-learn integration
python run_dges_causallearn.py
```


## Acknowledgements

We would like to sincerely thank these related works and open-sourced codes which our work is based on:

- Causal-learn package: [https://github.com/py-why/causal-learn](https://github.com/py-why/causal-learn).
- GES implementation: [https://github.com/juangamella/ges](https://github.com/juangamella/ges).
