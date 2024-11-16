import numpy as np
from csl.graph.AdjacencyConfusion import AdjacencyConfusion
from csl.graph.ArrowConfusion import ArrowConfusion
from csl.graph.GeneralGraph import GeneralGraph
from csl.graph.GraphNode import GraphNode
from csl.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io
from scipy.special import expit as sigmoid
import igraph as ig
import random
import torch
from csl.search.JuanBased import ges
import time 
import os
from csl.graph.Edge import Edge
from csl.graph.Endpoint import Endpoint
from csl.utils.DAG2CPDAG import dag2cpdag
from csl.utils.PDAG2DAG import pdag2dag
from csl.search.ScoreBased.ExactSearch import bic_exact_search
from csl.utils.data_utils import simulate_dag, set_random_seed
from csl.search.ConstraintBased.PC import pc

from causallearn.search.ScoreBased.GES import ges as causallearn_ges
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search as causallearn_bic_exact_search
from causallearn.utils.KCI.GaussianKernel import GaussianKernel
from causallearn.utils.KCI.Kernel import Kernel
from scipy import stats



def evaluate_result_non_dc(gt, est, dc, ndc):
    gt_list = []
    est_list = []
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i,j]==-1 and gt[j,i]==1:
                gt_list.append( (i,j,1) ) # directed edges
            if gt[i,j]==-1 and gt[j,i]==-1:
                if i<j: 
                    gt_list.append( (i,j,-1) ) # undirected edges
                else:
                    gt_list.append( (j,i,-1) ) # undirected edges

    for i in range(est.shape[0]):
        for j in range(est.shape[1]):
            if est[i,j]==-1 and est[j,i]==1:
                est_list.append( (i,j,1) ) # directed edges
            if est[i,j]==-1 and est[j,i]==-1:
                if i<j: 
                    est_list.append( (i,j,-1) ) # undirected edges
                else:
                    est_list.append( (j,i,-1) ) # undirected edges

    gt_set = set(gt_list)
    est_set = set(est_list)
    # print("gt_set: ", gt_set)
    # print("est_set:", est_set)

    # keep only the DC-NDC parts.
    for i,j,k in gt_set:
        if (i in dc) and (j in dc):
            gt_set = gt_set - {(i,j,k)}
        if (i in ndc) and (j in ndc):
            gt_set = gt_set - {(i,j,k)}    
    for i,j,k in est_set:
        if (i in dc) and (j in dc):
            est_set = est_set - {(i,j,k)}
        if (i in ndc) and (j in ndc):
            est_set = est_set - {(i,j,k)}

    tp = len(gt_set.intersection(est_set))
    # assert len(est_set)!=0
    if len(est_set)==0:
        return None 

    precision = tp / len(est_set)
    recall = tp / len(gt_set)
    # assert (precision + recall)!=0
    if (precision + recall)==0:
        return None 

    f1 = 2 * precision * recall / (precision + recall)

    missing_edge = gt_set - est_set
    extra_edge = est_set - gt_set
    shd = len(missing_edge) + len(extra_edge)  

    result = {"shd":shd, "f1":f1, "precision":precision, "recall":recall} # "Missing_edge":len(missing_edge), "Extra_edge":len(extra_edge)
    print("Results: ", result)
    return [shd, f1, precision, recall]



def evaluate_result(gt, est):
    gt_list = []
    est_list = []
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i,j]==-1 and gt[j,i]==1:
                gt_list.append( (i,j,1) ) # directed edges
            if gt[i,j]==-1 and gt[j,i]==-1:
                if i<j: 
                    gt_list.append( (i,j,-1) ) # undirected edges
                else:
                    gt_list.append( (j,i,-1) ) # undirected edges

    for i in range(est.shape[0]):
        for j in range(est.shape[1]):
            if est[i,j]==-1 and est[j,i]==1:
                est_list.append( (i,j,1) ) # directed edges
            if est[i,j]==-1 and est[j,i]==-1:
                if i<j: 
                    est_list.append( (i,j,-1) ) # undirected edges
                else:
                    est_list.append( (j,i,-1) ) # undirected edges

    gt_set = set(gt_list)
    est_set = set(est_list)
    # print("gt_set: ", gt_set)
    # print("est_set:", est_set)

    tp = len(gt_set.intersection(est_set))
    # assert len(est_set)!=0
    if len(est_set)==0:
        return None 

    precision = tp / len(est_set)
    recall = tp / len(gt_set)
    # assert (precision + recall)!=0
    if (precision + recall)==0:
        return None 

    f1 = 2 * precision * recall / (precision + recall)

    missing_edge = gt_set - est_set
    extra_edge = est_set - gt_set
    shd = len(missing_edge) + len(extra_edge)  

    result = {"shd":shd, "f1":f1, "precision":precision, "recall":recall} # "Missing_edge":len(missing_edge), "Extra_edge":len(extra_edge)
    print("Results: ", result)
    return [shd, f1, precision, recall]

# linear Gaussian model with determinisitc relation 
def simulate_linear_gaussian_deterministic(W, n):
    d = W.shape[0]
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    if not G.is_dag():
        raise ValueError('W must be a DAG')
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])

    b = np.random.uniform(low=1, high=3, size=(d,d))
    b = b * W

    low = 1
    high = 2

    deter_cluster = []
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        if len(parents)>1:
            choice = j
            deter_cluster = np.concatenate(([j], parents))
            # print(deter_cluster)
            break 
    print(f"This is Linear Gaussian Model. The order is: {np.array(ordered_vertices)}. The deterministic root: {[choice]}. The cluster: {deter_cluster}.")    

    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        if j==choice: # deterministic 
            var = np.random.uniform(low=low, high=high)
            eps = np.random.normal(0, var, size=n)
            X[:,j] = X[:,parents] @ b[parents,j]

        else: # non deterministic
            var = np.random.uniform(low=low, high=high)
            eps = np.random.normal(0, var, size=n)
            X[:,j] = X[:,parents] @ b[parents,j] + eps
    return X, choice 


# Nonlinear model with determinisitc relation 
def simulate_nonlinear_deterministic(W, n):
    def f(x):
        # det = np.random.randint(2) 
        # if det == 1:
        #     y = x**2
        # elif det == 2:
        #     y = np.tanh(x)

        y = np.tanh(x)
        # y = np.sin(x)

        return y 
    
    d = W.shape[0]
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    if not G.is_dag():
        raise ValueError('W must be a DAG')
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])

    b = np.random.uniform(low=1, high=3, size=(d,d))
    b = b * W

    low = 1
    high = 2

    deter_cluster = []
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        if len(parents)>1:
            choice = j
            deter_cluster = np.concatenate(([j], parents))
            # print(deter_cluster)
            break 
    print(f"This is Non-Linear Model. The order is: {np.array(ordered_vertices)}. The deterministic root: {[choice]}. The cluster: {deter_cluster}.")    
    non_det_cluster = list(set(list(range(d))) - set(deter_cluster))
    # print(non_det_cluster)

    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        if j==choice: # deterministic 
            var = np.random.uniform(low=low, high=high)
            eps = np.random.normal(0, var, size=n)
            X[:,j] = f(X[:,parents]) @ b[parents,j]

        else: # non deterministic
            var = np.random.uniform(low=low, high=high)
            if np.random.rand()>0.5:
                eps = np.random.uniform(low=-0.5, high=0.5, size=n)
            else:
                eps = np.random.normal(0, var, size=n)
            X[:,j] = f(X[:,parents]) @ b[parents,j] + eps
    return X, choice, deter_cluster, non_det_cluster


def generate_graph_linear(A, D, directory):
    # Get ground truth graphs
    directed_edges_gt = {(i, j) for i in range(A.shape[0]) for j in range(A.shape[0]) if A[i,j] != 0 and A[j,i]==0}  
    nodes = [GraphNode(f'X{i}') for i in range(A.shape[0])]   
    general_graph = GeneralGraph(nodes=nodes)
    
    for i, j in directed_edges_gt: general_graph.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW))
    pyd = GraphUtils.to_pydot(general_graph)
    pyd.write_png(f'{directory}/gt_dag.png')
    
    cpdag_gt = dag2cpdag(general_graph)
    pyd = GraphUtils.to_pydot(cpdag_gt)
    pyd.write_png(f'{directory}/gt_cpdag.png')
    
    # Get estimation graphs
    directed_edges_est = {(i, j) for i in range(D.shape[0]) for j in range(D.shape[0]) if D[i,j] != 0 and D[j,i]==0}  
    undirected_edges_est = {(i, j) for i in range(D.shape[0]) for j in range(i + 1, D.shape[0]) if D[j, i] != 0 and D[i, j]!= 0} 
    all_edges = general_graph.get_graph_edges()
    general_graph.remove_edges(all_edges)
    for i, j in directed_edges_est: general_graph.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW))
    for i, j in undirected_edges_est: general_graph.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))
    pyd = GraphUtils.to_pydot(general_graph)
    pyd.write_png(f'{directory}/est_pdag.png')
    
    dag_est = pdag2dag(general_graph)
    cpdag_est = dag2cpdag(dag_est)
    pyd = GraphUtils.to_pydot(cpdag_est)
    pyd.write_png(f'{directory}/est_cpdag.png')
    return cpdag_gt.graph, cpdag_est.graph

def generate_graph_nonlinear(A, D, directory):
    # Get ground truth graphs
    directed_edges_gt = {(i, j) for i in range(A.shape[0]) for j in range(A.shape[0]) if A[i,j] != 0 and A[j,i]==0}  
    nodes = [GraphNode(f'X{i}') for i in range(A.shape[0])]   
    general_graph = GeneralGraph(nodes=nodes)
    
    for i, j in directed_edges_gt: general_graph.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW))
    pyd = GraphUtils.to_pydot(general_graph)
    pyd.write_png(f'{directory}/gt_dag.png')
    
    cpdag_gt = dag2cpdag(general_graph)
    pyd = GraphUtils.to_pydot(cpdag_gt)
    pyd.write_png(f'{directory}/gt_cpdag.png')
    
    # Get estimation graphs
    directed_edges_est = {(i, j) for i in range(D.shape[0]) for j in range(D.shape[0]) if D[i,j] == -1 and D[j,i]==1}  
    undirected_edges_est = {(i, j) for i in range(D.shape[0]) for j in range(i + 1, D.shape[0]) if D[j, i]==-1 and D[i, j]==-1} 
    all_edges = general_graph.get_graph_edges()
    general_graph.remove_edges(all_edges)
    for i, j in directed_edges_est: general_graph.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW))
    for i, j in undirected_edges_est: general_graph.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))
    pyd = GraphUtils.to_pydot(general_graph)
    pyd.write_png(f'{directory}/est_pdag.png')
    
    dag_est = pdag2dag(general_graph)
    cpdag_est = dag2cpdag(dag_est)
    pyd = GraphUtils.to_pydot(cpdag_est)
    pyd.write_png(f'{directory}/est_cpdag.png')
    return cpdag_gt.graph, cpdag_est.graph

def generate_graph_gt(A, directory):
    # Get ground truth graphs
    directed_edges_gt = {(i, j) for i in range(A.shape[0]) for j in range(A.shape[0]) if A[i,j] != 0 and A[j,i]==0}  
    # undirected_edges_gt = {(i, j) for i in range(A.shape[0]) for j in range(i + 1, A.shape[0]) if A[j, i] != 0 and A[i, j] != 0} 
    nodes = [GraphNode(f'X{i}') for i in range(A.shape[0])]   
    general_graph = GeneralGraph(nodes=nodes)
    
    for i, j in directed_edges_gt: general_graph.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW))
    # for i, j in undirected_edges_gt: general_graph.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))
    pyd = GraphUtils.to_pydot(general_graph)
    pyd.write_png(f'{directory}/test_dag.png')
    
    # general_graph = pdag2dag(general_graph)
    cpdag_gt = dag2cpdag(general_graph)
    pyd = GraphUtils.to_pydot(cpdag_gt)
    pyd.write_png(f'{directory}/test_cpdag.png')
    return cpdag_gt.graph

# def check_loop(D):
#     for i in range(D.shape[0]):
#         for j in range(i+1, D.shape[0]):
#             if D[j, i] != 0 and D[i, j]!= 0:
#                 D[j, i] = 0
#                 D[i, j] = 0
#     G = ig.Graph.Weighted_Adjacency(D.tolist())
#     if not G.is_dag():
#         raise ValueError('D must be a DAG')

def run_linear_methods(seed=0, n=1000, d=6, method='dges'):
    # n = 10000
    # d = 6
    # method = 'pc'
    set_random_seed(seed)
    print(f"This is Linear instance {[seed]}: variable: {d}, samples: {n}, method: {method}.")

    A = simulate_dag(d, d, graph_type='ER') # binary matrix
    X, _ = simulate_linear_gaussian_deterministic(A, n) # data matrix X_{n*d}
    
    ########################## Ours: DGES ###############################################
    if method=='dges':
        directory = f"simulate_deterministic/linear/dges/dges_{d}_{n}/seed_{seed}/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Step 1: run GES.
        B, cp_dag_est, score = ges.fit_bic(X, seed=seed, directory=directory)

        # Step 2: find deterministic clusters and their neighbors.
        # 2.1. Find deterministic cluster.
        X = X - np.mean(X, axis=0) # centering data
        for i in range(d):
            parents = np.where(B[:,i]!=0)[0]
            # print(f"The variale: {i}, the parents: {parents}.") 
            if len(parents)>0:
                y_i = X[:,i]        
                x_i = np.atleast_2d(X[:,parents])
                # print(y_i.shape, x_i.shape)
                coef = np.linalg.lstsq(x_i, y_i, rcond=None)[0]
                sigma = np.var(y_i - x_i @ coef)
                # print(f"The coef: {coef}, the sigma is: {sigma}")
                if sigma < 1e-5:
                    det_cluster_index = np.where( abs(coef)>1e-5 )
                    # print(f"The deterministic cluster is {deter_cluster}")
                    det_cluster = np.concatenate(([i], parents[det_cluster_index]))
                    print(f"The estimated deterministic cluster is: {det_cluster}.")
                    # det_cluster_set = det_cluster_set.union( set(det_cluster) )
        assert len(det_cluster)>0

        # 2.2. Find deterministic cluster and its neighbors.
        all_neighbors = set()
        for i in range(d):
            if i in det_cluster:
                neighbors = np.where((B+B.T)[:,i]!=0)[0]
                all_neighbors = all_neighbors.union( set(neighbors) )
        all_neighbors = sorted(list(all_neighbors))
        print(f"The deterministic cluster and their neighbors: {all_neighbors}.")

        # Step 3: Run exact search. 
        X_new = X[:,all_neighbors]
        C, _ = bic_exact_search(X_new, super_graph=None, search_method='astar')
    
        # Step 4: Combine the results from GES and Exact Search. 
        D = np.zeros((d,d))
        non_neighbor = set(range(d)) - set(all_neighbors)
        print(f"The non-neighbors: {list(non_neighbor)}.")
        for i in non_neighbor:
            D[i,:] = B[i,:]
            D[:,i] = B[:,i]
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                if C[i,j] == 1:
                    D[all_neighbors[i], all_neighbors[j]] = 1
        
        # check_loop(D)
        cpdag_gt, cpdag_est = generate_graph(A, D, directory)

    ########################## Greedy Search ###############################################
    elif method=='ges':
        directory = f"simulate_deterministic/linear/ges/ges_{d}_{n}/seed_{seed}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        B, cp_dag_est, score = ges.fit_bic(X, seed=seed, directory=directory)
        cpdag_gt, cpdag_est = generate_graph(A, B, directory)

    ########################## Exact Search - ASTAR ###############################################
    elif method=='astar':
        directory = f"simulate_deterministic/linear/astar/astar_{d}_{n}/seed_{seed}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        B, _ = bic_exact_search(X, super_graph=None, search_method='astar')
        cpdag_gt, cpdag_est = generate_graph(A, B, directory)
    
    elif method=='pc':
        directory = f"simulate_deterministic/linear/pc/pc_{d}_{n}/seed_{seed}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        from csl.utils.cit import fisherz
        cg = pc(X, 0.05, fisherz)
        cpdag_gt, cpdag_est = generate_graph(A, cg.G.graph, directory)

    else:
        raise ValueError("Unknown search method.")
     
    ########################## Evaluation ############################################### 
    res_list = evaluate_result(cpdag_gt, cpdag_est)
    return res_list
 

def check_dag(A, dc, ndc):
    dc = np.array(dc)
    ndc = np.array(ndc)
    a, b = A.shape
    dc_ndc_list = []
    for i in range(a):
        for j in range(b):
            if A[i,j]==1:
                if (i in dc) and (j in ndc):
                    dc_ndc_list.append( (i,j) )
                if (i in ndc) and (j in dc):
                    dc_ndc_list.append( (i,j) )
    # if len(dc_ndc_list)
    flag = False 
    print( len(dc_ndc_list) )
    if len(dc_ndc_list) < 1:
        flag=True 
    for i in ndc:
        if np.sum(A[i,dc]) >= len(dc) - 1:
            flag = True
            print(i)
        if np.sum(A[dc,i]) >= len(dc) - 1: 
            flag = True
            print(i)

    print(f"The flag: {flag}.")
    return flag



def run_nonlinear_methods(seed=0, n=1000, d=6, method='dges'):
    # n = 10000
    # d = 6
    # method = 'pc'
    set_random_seed(seed)
    print(f"This is Nonlinear instance {[seed]}: variable: {d}, samples: {n}, method: {method}.")

    s0 = d #int(d*1.5)
    A = simulate_dag(d, s0, graph_type='ER') # binary matrix

    X, _, dc, ndc = simulate_nonlinear_deterministic(A, n) # data matrix X_{n*d}
    
    flag = check_dag(A, dc, ndc)
    if flag:
        return None

    
    ########################## Ours: DGES ###############################################
    if method=='dges':
        directory = f"simulate_deterministic/nonlinear/dges/dges_{d}_{n}/seed_{seed}/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Step 1: run GES.
        # B, cp_dag_est, score = ges.fit_bic(X, seed=seed, directory=directory)
        cg = causallearn_ges(X, score_func='local_score_CV_general')
        B = cg['G'].graph

        # Step 2: find deterministic clusters and their neighbors.
        # 2.1. Find deterministic cluster.
        det_cluster = []
        # det_cluster_set = set()
        X = X - np.mean(X, axis=0) # centering data
        for i in range(d):
            parents = np.where(B[:,i]!=0)[0]
            # print(f"The variale: {i}, the parents: {parents}.") 
            if len(parents)>0:
                # # Linear cases: x -> y.
                # y_i = X[:,i].reshape(-1,1)      
                # x_i = np.atleast_2d(X[:,parents])
                # coef = np.linalg.lstsq(x_i, y_i, rcond=None)[0]
                # sigma = np.var(y_i - x_i @ coef)
                # if sigma < 1e-5:
                #     det_cluster_index = np.where( abs(coef)>1e-5 )
                #     # print(f"The deterministic cluster is {deter_cluster}")
                #     det_cluster = np.concatenate(([i], parents[det_cluster_index]))
                #     print(f"The estimated deterministic cluster is: {det_cluster}.")
                
                # Nonlinear cases: x -> y.
                y_i = X[:,i].reshape(-1,1)        
                x_i = np.atleast_2d(X[:,parents])
                # print(y_i.shape, x_i.shape)
                data_x = stats.zscore(x_i, ddof=1, axis=0) # normalize data
                data_y = stats.zscore(y_i, ddof=1, axis=0)

                kernelX = GaussianKernel()
                kernelX.set_width_empirical_kci(data_x)
                Kx = kernelX.kernel(data_x)
                Kx = Kernel.center_kernel_matrix(Kx) # centering kernel matrix

                kernelY = GaussianKernel()
                kernelY.set_width_empirical_kci(data_y)
                Ky = kernelY.kernel(data_y)
                Ky = Kernel.center_kernel_matrix(Ky)
                
                epsilon = 1e-8
                R = epsilon * np.linalg.inv(Kx + np.eye(n) * epsilon)
                K_R = R @ Ky @ R
                stat = np.trace(K_R @ K_R)
                stat = 1/n * np.sqrt(stat)
                print(f"The stat for variable {i} is: {stat}.")

                if stat < 1e-3:
                    # det_cluster_index = np.where( abs(coef)>1e-5 )
                    # print(f"The deterministic cluster is {deter_cluster}")
                    det_cluster = np.concatenate(([i], parents))
                    print(f"The estimated deterministic cluster is: {det_cluster}.")
                    # det_cluster_set = det_cluster_set.union( set(det_cluster) )
                    break 
        # assert len(det_cluster)>0
        if len(det_cluster)==0:
            print("Warning: Cannot find deterministic cluster.")
            cpdag_gt, cpdag_est = generate_graph_nonlinear(A, B, directory)

        else:
            # 2.2. Find deterministic cluster and its neighbors.
            all_neighbors = set()
            for i in range(d):
                if i in det_cluster:
                    neighbors = np.where(B[:,i]!=0)[0]
                    all_neighbors = all_neighbors.union( set(neighbors) )
            all_neighbors = sorted(list(all_neighbors))
            print(f"The deterministic cluster and their neighbors: {all_neighbors}.")

            # Step 3: Run exact search. 
            X_new = X[:,all_neighbors]
            # C, _ = bic_exact_search(X_new, super_graph=None, search_method='astar')
            C, _ = causallearn_bic_exact_search(X_new, super_graph=None, search_method='astar')
        
            # Step 4: Combine the results from GES and Exact Search. 
            D = np.zeros((d,d))
            non_neighbor = set(range(d)) - set(all_neighbors)
            print(f"The non-neighbors: {list(non_neighbor)}.")
            for i in non_neighbor:
                D[i,:] = B[i,:]
                D[:,i] = B[:,i]
            for i in range(C.shape[0]):
                for j in range(C.shape[1]):
                    if C[i,j] == 1:
                        D[all_neighbors[i], all_neighbors[j]] = -1
                        D[all_neighbors[j], all_neighbors[i]] = 1
                        # D[i,j] == -1 and D[j,i]==1
            
            # check_loop(D)
            cpdag_gt, cpdag_est = generate_graph_nonlinear(A, D, directory)

    ########################## Greedy Search ###############################################
    elif method=='ges':
        directory = f"simulate_deterministic/nonlinear/ges/ges_{d}_{s0}_{n}/seed_{seed}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        # B, cp_dag_est, score = ges.fit_bic(X, seed=seed, directory=directory)
        cg = causallearn_ges(X, score_func='local_score_CV_general')  # [local_score_CV_general, local_score_marginal_general]
        cpdag_gt, cpdag_est = generate_graph_nonlinear(A, cg['G'].graph, directory)

    ########################## Exact Search - ASTAR ###############################################
    elif method=='astar':
        directory = f"simulate_deterministic/nonlinear/astar/astar_{d}_{d+2}_{n}/seed_{seed}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        B, _ = causallearn_bic_exact_search(X, super_graph=None, search_method='astar')
        cpdag_gt, cpdag_est = generate_graph_linear(A, B, directory)
    
    elif method=='dp':
        directory = f"simulate_deterministic/nonlinear/dp/dp_{d}_{n}/seed_{seed}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        B, _ = causallearn_bic_exact_search(X, super_graph=None, search_method='dp')
        cpdag_gt, cpdag_est = generate_graph_linear(A, B, directory)

    elif method=='pc':
        directory = f"simulate_deterministic/nonlinear/pc/pc_{d}_{n}/seed_{seed}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        # from csl.utils.cit import fisherz
        from csl.utils.cit import kci
        cg = pc(X, 0.05, kci)
        cpdag_gt, cpdag_est = generate_graph_nonlinear(A, cg.G.graph, directory)

    else:
        raise ValueError("Unknown search method.")
     
    ########################## Evaluation ############################################### 
    # res_list = evaluate_result(cpdag_gt, cpdag_est)
    res_list = evaluate_result_non_dc(cpdag_gt, cpdag_est, dc, ndc)
    return res_list
 

if __name__=='__main__':
    res_matrix = []
    seed_start = 31
    count = 0

    n = 100          # sample size: [100, 150, 200, 250, 300].
    d = 8            # variable: [6, 7, 8, 9, 10, 11].
    method = 'astar'  # methods: ['dges', 'astar', 'ges', 'pc'].
    path = f"simulate_deterministic/nonlinear/result_all.csv"
    f = open(path, 'a+')
    if not os.path.getsize(path): # create the first line for the first time.
        f.write(f"method,n,d,random_seed,SHD,F1,precison,recall,runtime\n")
    for i in range(seed_start, seed_start+100):
        start = time.time()
        linear = False    # linear or nonlinear.
        
        # res = run_linear_methods(i, n, d, method) 
        res = run_nonlinear_methods(i, n, d, method) 
        end = time.time()
        print(f"The time cost is: {end-start}s.\n")
        if res is not None:
            res.append(end-start)
            res_matrix.append(res)
            f.write(f"{method},{n},{d},{i},{res[0]},{res[1]},{res[2]},{res[3]},{res[4]}\n")
            count += 1
        if count>=30:
            break

    res_matrix = np.array(res_matrix)
    print(f"SHD | F1 | Precision | Recall | Time.")
    print("Mean: ", np.mean(res_matrix, axis=0))
    print('Std: ', np.std(res_matrix, axis=0))
    f.close()

    path = f"simulate_deterministic/nonlinear/result_avg.csv"
    f_avg = open(path, 'a+')
    if not os.path.getsize(path): # create the first line for the first time.
        f_avg.write(f"method,n,d,SHD_mean,F1_mean,precison_mean,recall_mean,runtime_mean,SHD_std,F1_std,precison_std,recall_std,runtime_std\n")
    my_mean = np.mean(res_matrix, axis=0)
    my_std = np.std(res_matrix, axis=0)
    f_avg.write(f"{method},{n},{d},{my_mean[0]},{my_mean[1]},{my_mean[2]},{my_mean[3]},{my_mean[4]},{my_std[0]},{my_std[1]},{my_std[2]},{my_std[3]},{my_std[4]}\n")
    f_avg.close()
    