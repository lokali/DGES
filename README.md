# [NeurIPS 2024] On Causal Discovery in the Presence of Deterministic Relations

This is the implementation of synthetic experiments for the paper **"[On Causal Discovery in the Presence of Deterministic Relations](https://openreview.net/forum?id=pfvcsgFrJ6)"**, NeurIPS 2024.

## Overview

- We find that **score-based methods with exact search** can naturally address the issues of deterministic relations, under mild assumptions.
- To improve efficiency, we propose the novel and **versatile framework** called Determinism-aware Greedy Equivalent Search (DGES).
- We provide the **identifiability conditions** of DGES under general functional models.

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

## How to Use

### 1. Environment Installation

```sh
pip install causal-learn
```

### 2. Quick Start

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

### 3. More Usage Details

#### 3.1 Fast Version (with only Phase 1 + 2, fast)

```python
import numpy as np
from causallearn.search.ScoreBased.DGES import dges

# X: data matrix with shape (n_samples, n_features)
Record = dges(X)

# Access results
G = Record['G']                    # learned CPDAG (GeneralGraph)
mindcs = Record['mindcs']          # detected MinDCs
score = Record['score']            # final score
```

#### 3.2 Full Version (with 3 Phases including Exact Search)

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

#### 3.3 Usage of Nonlinear Score Functions

DGES also supports nonlinear score functions for data with nonlinear causal relationships:

```python
Record = dges(X,
              score_func='local_score_CV_general',
              det_threshold=0.999)   # R^2 threshold for nonlinear DC detection
```

#### 3.4 Other Score Function Options

| Score Function                               | Data Type             | Description                             |
| -------------------------------------------- | --------------------- | --------------------------------------- |
| `'local_score_BIC_from_cov_deterministic'` | Linear, continuous    | Modified BIC with epsilon (recommended) |
| `'local_score_BIC_from_cov'`               | Linear, continuous    | Standard BIC (without DC handling)      |
| `'local_score_CV_general'`                 | Nonlinear, continuous | Cross-validated kernel score            |
| `'local_score_marginal_general'`           | Nonlinear, continuous | Marginal likelihood kernel score        |
| `'local_score_BDeu'`                       | Discrete              | BDeu score                              |

## Acknowledgements

We would like to sincerely thank these related works and open-sourced codes which our work is based on:

- Causal-learn package: [https://github.com/py-why/causal-learn](https://github.com/py-why/causal-learn).
- GES implementation: [https://github.com/juangamella/ges](https://github.com/juangamella/ges).

If you find our work useful, please consider citing:

```
@inproceedings{li2024causal,
  title={On Causal Discovery in the Presence of Deterministic Relations},
  author={Li, Loka and Dai, Haoyue and Al Ghothani, Hanin and Huang, Biwei and Zhang, Jiji and Harel, Shahar and Bentwich, Isaac and Chen, Guangyi and Zhang, Kun},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
  year={2024}
}
```
