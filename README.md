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

## DGES and Implementations

- **DGES in theory** comprises three phases: 
    - (1) identify minimal deterministic clusters (i.e., the minimal set of variables with deterministic relationships); 
    - (2) run modified Greedy Equivalent Search (GES) to obtain an initial graph, and 
    - (3) perform exact search exclusively on each deterministic cluster and its neighbors. 

- **Modified GES** mainly has two changes:

    - Modified BIC score functions: 
        - Before: $\log L \propto -\frac{n}{2}(1+ \log |\Sigma|)$  
        - After: $\log L \propto -\frac{n}{2}(1+ \log |\Sigma+\epsilon|)$; 

    - Modified edge adding and deleting.

- **Efficient DGES in Implementation**: As you may notice, the first phase could be time-consuming, therefore, we update our code (in the current github version) by: exchanging the first and second phases, so that the search space is smaller for detect MinDCs. 

- **Another Efficient Implementation**: If you are running a large-scale graph, then you may just use GES + modified BIC score, to quickly get an approximate solution. You can skip the MinDC-detection phase and exact-search phase, only running the modified-GES phase without modified-edge-adding-and-deleting step. 


## How to Run

- Installation: 

```sh
# Set up a new conda environment with Python 3.8.
conda create -n DGES python=3.8
conda activate DGES

# Install necessary causal-learn.
pip install causal-learn

# Install other python libraries.
pip install numpy pydot networkx 
```

- Quick start:
```sh
# Parameters:
#     method: which method to run
#     count:  number of instances to evaluate
#     d: number of variables
#     n: number of samples 
python run_dges_linear_1DC.py  # Linear model with only one minimal deterministic cluster (MinDC) 
python run_dges_linear_2DC.py  # Linear model with two minimal deterministic clusters (MinDCs) 
python run_dges_nonlinear_1DC.py # Nonlinear model with only one minimal deterministic cluster (MinDC)
python run_dges_nonlinear_2DC.py # Nonlinear model with two minimal deterministic clusters (MinDCs) 
```

## Acknowledgements


We would like to sincerely thank these related works and open-sourced codes which our work is based on:

- Causal-learn package: [https://github.com/py-why/causal-learn](https://github.com/py-why/causal-learn).
- GES implementation: [https://github.com/juangamella/ges](https://github.com/juangamella/ges).
