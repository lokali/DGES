"""
Example: Running DGES via causal-learn package.

DGES is now integrated into the causal-learn package as:
    causallearn.search.ScoreBased.DGES.dges

This script demonstrates both linear and nonlinear usage with
deterministic relations.
"""

import numpy as np
import igraph as ig


def set_random_seed(seed):
    np.random.seed(seed)


def simulate_dag(d, s, graph_type='ER'):
    """Generate a random DAG adjacency matrix."""
    if graph_type == 'ER':
        prob = s / (d * (d - 1) / 2)
        B = np.tril(np.random.binomial(1, prob, size=(d, d)), k=-1)
        perm = np.random.permutation(d)
        B = B[perm][:, perm]
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")
    return B


def simulate_linear_gaussian_deterministic(W, n, num_dc=1):
    """
    Simulate linear Gaussian data with deterministic relations.

    For nodes with >1 parent, the first `num_dc` such nodes are made
    deterministic (zero noise variance).
    """
    d = W.shape[0]
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    if not G.is_dag():
        raise ValueError('W must be a DAG')

    ordered_vertices = G.topological_sorting()
    X = np.zeros([n, d])

    # Random edge weights
    b = np.random.uniform(low=1, high=3, size=(d, d))
    b = b * W

    # Find nodes to make deterministic
    det_roots = []
    det_cluster = set()
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        if len(parents) > 1:
            det_roots.append(j)
            det_cluster.update([j] + parents)
            if len(det_roots) >= num_dc:
                break

    non_det = list(set(range(d)) - det_cluster)

    # Simulate data
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        if j in det_roots:
            # Deterministic: no noise
            X[:, j] = X[:, parents] @ b[parents, j]
        else:
            # Stochastic: add Gaussian noise
            var = np.random.uniform(low=1, high=2)
            X[:, j] = X[:, parents] @ b[parents, j] + np.random.normal(0, var, size=n)

    return X, det_roots, sorted(det_cluster), non_det


def run_dges_example(seed=0, n=1000, d=8, num_dc=1):
    """Run DGES on simulated data with deterministic relations."""
    from causallearn.search.ScoreBased.DGES import dges
    from causallearn.graph.GeneralGraph import GeneralGraph
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint
    from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
    from causallearn.utils.DAG2CPDAG import dag2cpdag

    set_random_seed(seed)
    print(f"=== DGES Example: seed={seed}, d={d}, n={n}, num_dc={num_dc} ===")

    # Generate DAG and data
    A = simulate_dag(d, d, graph_type='ER')
    X, det_roots, dc, ndc = simulate_linear_gaussian_deterministic(A, n, num_dc)

    if len(det_roots) == 0:
        print("No deterministic relations found in this random graph. Skipping.")
        return None

    print(f"Deterministic roots: {det_roots}")
    print(f"Deterministic cluster: {dc}")

    # Ground truth CPDAG
    nodes = [GraphNode(f'X{i}') for i in range(d)]
    G_true = GeneralGraph(nodes)
    for i in range(d):
        for j in range(d):
            if A[i, j] != 0:
                G_true.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW))
    G_true = dag2cpdag(G_true)

    # Run DGES (Phase 1 + 2 only, fast)
    Record = dges(X, score_func='local_score_BIC_from_cov_deterministic',
                  det_threshold=1e-5, det_epsilon=0.01,
                  skip_exact_search=True)

    G_est = Record['G']
    mindcs = Record.get('mindcs', [])
    print(f"Detected MinDCs: {mindcs}")

    # Evaluate
    adj = AdjacencyConfusion(G_true, G_est)
    ap = adj.get_adj_precision()
    ar = adj.get_adj_recall()
    print(f"Adjacency Precision: {ap:.3f}")
    print(f"Adjacency Recall:    {ar:.3f}")
    print()

    return {'ap': ap, 'ar': ar}


def run_dges_with_exact_search(seed=0, n=1000, d=6, num_dc=1):
    """Run full 3-phase DGES (including exact search)."""
    from causallearn.search.ScoreBased.DGES import dges
    from causallearn.graph.GeneralGraph import GeneralGraph
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint
    from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
    from causallearn.utils.DAG2CPDAG import dag2cpdag

    set_random_seed(seed)
    print(f"=== DGES Full (with Exact Search): seed={seed}, d={d}, n={n} ===")

    A = simulate_dag(d, d, graph_type='ER')
    X, det_roots, dc, ndc = simulate_linear_gaussian_deterministic(A, n, num_dc)

    if len(det_roots) == 0:
        print("No deterministic relations found. Skipping.")
        return None

    print(f"Deterministic roots: {det_roots}")
    print(f"Deterministic cluster: {dc}")

    # Ground truth CPDAG
    nodes = [GraphNode(f'X{i}') for i in range(d)]
    G_true = GeneralGraph(nodes)
    for i in range(d):
        for j in range(d):
            if A[i, j] != 0:
                G_true.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW))
    G_true = dag2cpdag(G_true)

    # Run full DGES (Phase 1 + 2 + 3)
    Record = dges(X, score_func='local_score_BIC_from_cov_deterministic',
                  det_threshold=1e-5, det_epsilon=0.01,
                  skip_exact_search=False,
                  exact_search_method='astar')

    G_est = Record['G']
    exact_nodes = Record.get('exact_search_nodes', [])
    print(f"Exact search ran on nodes: {exact_nodes}")

    adj = AdjacencyConfusion(G_true, G_est)
    ap = adj.get_adj_precision()
    ar = adj.get_adj_recall()
    print(f"Adjacency Precision: {ap:.3f}")
    print(f"Adjacency Recall:    {ar:.3f}")
    print()

    return {'ap': ap, 'ar': ar}


if __name__ == '__main__':
    import time

    # Example 1: Fast DGES (Phase 1 + 2 only)
    print("=" * 60)
    print("Example 1: DGES (Phase 1 + 2, skip exact search)")
    print("=" * 60)
    results = []
    for seed in range(5):
        start = time.time()
        res = run_dges_example(seed=seed, n=1000, d=8, num_dc=1)
        elapsed = time.time() - start
        if res is not None:
            res['time'] = elapsed
            results.append(res)
            print(f"Time: {elapsed:.2f}s\n")

    if results:
        print("--- Average Results (Phase 1+2) ---")
        print(f"AP: {np.mean([r['ap'] for r in results]):.3f} +/- {np.std([r['ap'] for r in results]):.3f}")
        print(f"AR: {np.mean([r['ar'] for r in results]):.3f} +/- {np.std([r['ar'] for r in results]):.3f}")
        print()

    # Example 2: Full DGES with exact search (smaller graph)
    print("=" * 60)
    print("Example 2: Full DGES (Phase 1 + 2 + 3, with exact search)")
    print("=" * 60)
    results_full = []
    for seed in range(3):
        start = time.time()
        res = run_dges_with_exact_search(seed=seed, n=1000, d=6, num_dc=1)
        elapsed = time.time() - start
        if res is not None:
            res['time'] = elapsed
            results_full.append(res)
            print(f"Time: {elapsed:.2f}s\n")

    if results_full:
        print("--- Average Results (Full DGES) ---")
        print(f"AP: {np.mean([r['ap'] for r in results_full]):.3f} +/- {np.std([r['ap'] for r in results_full]):.3f}")
        print(f"AR: {np.mean([r['ar'] for r in results_full]):.3f} +/- {np.std([r['ar'] for r in results_full]):.3f}")
