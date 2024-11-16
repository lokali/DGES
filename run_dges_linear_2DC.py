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
    assert len(est_set)!=0
    precision = tp / len(est_set)
    recall = tp / len(gt_set)
    assert (precision + recall)!=0
    f1 = 2 * precision * recall / (precision + recall)

    missing_edge = gt_set - est_set
    extra_edge = est_set - gt_set
    shd = len(missing_edge) + len(extra_edge)  

    result = {"shd":shd, "f1":f1, "precision":precision, "recall":recall} # "Missing_edge":len(missing_edge), "Extra_edge":len(extra_edge)
    print("Results: ", result, '\n')
    return [shd, f1, precision, recall]



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
 
    choice = []        # deterministic root
    deter_cluster = [] # deterministic set
    non_det_cluster = [] # non-deterministic set 
    count = 0
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        if len(parents)>1 and len(parents)<4:   #[1,4]
            choice.append(j)
            deter_cluster.append(np.concatenate(([j], parents)))
            count += 1
        if count==2:
            break 
  
    flag = False 
    if count==1:
        non_det_cluster = list(set(list(range(d))) - set(deter_cluster[0]))
    if count==2:
        non_det_cluster = list(set(list(range(d))) - set(deter_cluster[0]) - set(deter_cluster[1])) 
    if count==0:
        flag = True 
    if len(deter_cluster)==0 or len(non_det_cluster)==0:
        flag = True         

    print(f"This is Linear Gaussian Model. The order is: {np.array(ordered_vertices)}. The deterministic root: {[choice]}. The cluster: {deter_cluster}. Non-cluster: {non_det_cluster}.")   
      

    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        if j in choice: # deterministic 
            var = np.random.uniform(low=low, high=high)
            eps = np.random.normal(0, var, size=n)
            X[:,j] = X[:,parents] @ b[parents,j]

        else: # non deterministic
            var = np.random.uniform(low=low, high=high)
            eps = np.random.normal(0, var, size=n)
            X[:,j] = X[:,parents] @ b[parents,j] + eps 
    if len(deter_cluster)>1:
        deter_cluster = list(set(deter_cluster[0]).union(set(deter_cluster[1])))
    return X, choice, deter_cluster, non_det_cluster, flag    

def generate_graph(A, D, directory):
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
    print(f"The number of DC-NDC edges:", len(dc_ndc_list) )
    if len(dc_ndc_list) < 1:
        flag=True 
    for i in ndc:
        if np.sum(A[i,dc]) >= len(dc) - 1:
            flag = True
            # print(i)
        if np.sum(A[dc,i]) >= len(dc) - 1: 
            flag = True
            # print(i)

    print(f"The flag: {flag}.")
    return flag


def run_linear_methods(seed=0, n=1000, d=6, method='dges'):
    # n = 10000
    # d = 6
    # method = 'pc'
    set_random_seed(seed)
    print(f"This is Linear instance {[seed]}: variable: {d}, samples: {n}, method: {method}.")

    A = simulate_dag(d, d, graph_type='ER') # binary matrix
    X, _, dc, ndc, flag = simulate_linear_gaussian_deterministic(A, n) # data matrix X_{n*d}

    # print(_, dc, ndc)

    flag = check_dag(A, dc, ndc)
    # print(flag) 

    if flag:
        return None
    
    ########################## Ours: DGES ###############################################
    if method=='dges':
        directory = f"simulate_deterministic_overlap/linear/dges/dges_{d}_{n}/seed_{seed}/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Step 1: run GES.
        B, cp_dag_est, score = ges.fit_bic(X, seed=seed, directory=directory)

        # Step 2: find deterministic clusters and their neighbors.
        # 2.1. Find deterministic cluster.
        det_clusters = set()     # e.g.:  {1, 2, 5, 6}
        det_clusters_list = []   # e.g.:  [{1, 6}, {2, 5}]
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
                    det_cluster = set(sorted(det_cluster))
                    print(f"The estimated deterministic cluster is: {det_cluster}.")
                    if det_cluster not in det_clusters_list:
                        det_clusters_list.append(det_cluster)
                    # det_clusters = det_clusters.union( set(det_cluster) )
                    det_clusters.update(det_cluster) 

        # print(det_clusters, det_clusters_list)
        assert len(det_cluster)>0
        print(f"The whole det clusters are: {det_clusters}. | There are {len(det_clusters_list)} clusters, and the separate list is: {det_clusters_list}.")

        # 2.2. Find deterministic cluster and its neighbors.
        all_neighbors = set()
        for i in det_clusters:
            neighbors = np.where((B+B.T)[:,i]!=0)[0]
            all_neighbors = all_neighbors.union( set(neighbors) )
            # print(f"neighbors: {neighbors}")
        extended_set = all_neighbors.union(det_clusters)
        extended_set = sorted(list(extended_set))
        print(f"The deterministic cluster and their neighbors: {extended_set} = {all_neighbors} U {det_clusters}")

        # Step 3: Run exact search. 
        X_new = X[:,extended_set]
        C, _ = bic_exact_search(X_new, super_graph=None, search_method='astar')
    
        # Step 4: Combine the results from GES and Exact Search. 
        D = np.zeros((d,d))
        non_neighbor = set(range(d)) - set(extended_set)
        print(f"The non-neighbors: {list(non_neighbor)}.")
        for i in non_neighbor:
            D[i,:] = B[i,:]
            D[:,i] = B[:,i]
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                if C[i,j] == 1:
                    D[extended_set[i], extended_set[j]] = 1
        
        # check_loop(D)
        cpdag_gt, cpdag_est = generate_graph(A, D, directory)

    ########################## Greedy Search ###############################################
    elif method=='ges':
        directory = f"simulate_deterministic_overlap/linear/ges/ges_{d}_{n}/seed_{seed}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        B, cp_dag_est, score = ges.fit_bic(X, seed=seed, directory=directory)
        cpdag_gt, cpdag_est = generate_graph(A, B, directory)

    ########################## Exact Search - ASTAR ###############################################
    elif method=='astar':
        directory = f"simulate_deterministic_overlap/linear/astar/astar_{d}_{n}/seed_{seed}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        B, _ = bic_exact_search(X, super_graph=None, search_method='astar')
        cpdag_gt, cpdag_est = generate_graph(A, B, directory)
    
    elif method=='pc':
        directory = f"simulate_deterministic_overlap/linear/pc/pc_{d}_{n}/seed_{seed}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        from csl.utils.cit import fisherz
        cg = pc(X, 0.05, fisherz)
        cpdag_gt, cpdag_est = generate_graph(A, cg.G.graph, directory)

    else:
        raise ValueError("Unknown search method.")
     
    ########################## Evaluation ############################################### 
    # res_list = evaluate_result(cpdag_gt, cpdag_est)
    res_list = evaluate_result_non_dc(cpdag_gt, cpdag_est, dc, ndc)
    return res_list
 
def run_nonlinear_methods(seed=0, n=1000, d=6, method='dges'):
    # n = 10000
    # d = 6
    # method = 'pc'
    set_random_seed(seed)
    print(f"This is Nonlinear instance {[seed]}: variable: {d}, samples: {n}, method: {method}.")

    A = simulate_dag(d, d, graph_type='ER') # binary matrix
    X, _ = simulate_linear_gaussian_deterministic(A, n) # data matrix X_{n*d}
    
    ########################## Ours: DGES ###############################################
    if method=='dges':
        directory = f"simulate_deterministic_overlap/nonlinear/dges/dges_{d}_{n}/seed_{seed}/"
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
        directory = f"simulate_deterministic_overlap/nonlinear/ges/ges_{d}_{n}/seed_{seed}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        # B, cp_dag_est, score = ges.fit_bic(X, seed=seed, directory=directory)
        cg = cl_ges(X, score_func='local_score_BIC')
        cpdag_gt, cpdag_est = generate_graph(A, cg.G.graph, directory)

    ########################## Exact Search - ASTAR ###############################################
    elif method=='astar':
        directory = f"simulate_deterministic_overlap/nonlinear/astar/astar_{d}_{n}/seed_{seed}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        B, _ = bic_exact_search(X, super_graph=None, search_method='astar')
        cpdag_gt, cpdag_est = generate_graph(A, B, directory)
    
    elif method=='pc':
        directory = f"simulate_deterministic_overlap/nonlinear/pc/pc_{d}_{n}/seed_{seed}/"
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

def main():
    res_matrix = []
    seed_start = 0
    count = 0

    n = 500          # sample size: [100, 250, 500, 1000, 2000].
    d = 8            # variable: [8, 10, 12, 14, 16, 18, 20].
    method = 'dges'  # methods: ['dges', 'astar', 'ges', 'pc'].
    directory = f"simulate_deterministic_overlap/linear"
    path = directory+'/result_all.csv'
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = open(path, 'a+')
    if not os.path.getsize(path): # create the first line for the first time.
        f.write(f"method,n,d,random_seed,SHD,F1,precison,recall,runtime\n")
    
    for i in range(seed_start, seed_start+100):
        start = time.time()
        
        res = run_linear_methods(i, n, d, method) 
        # res = run_nonlinear_methods(i, n, d, method) 
        end = time.time()
        print(f"The time cost is: {end-start}s.\n")
        if res is not None:
            res.append(end-start)
            res_matrix.append(res)
            f.write(f"{method},{n},{d},{i},{res[0]},{res[1]},{res[2]},{res[3]},{res[4]}\n")
            count += 1
        if count>=20:
            break

    res_matrix = np.array(res_matrix)
    print(f"SHD | F1 | Precision | Recall | Time.")
    print("Mean: ", np.mean(res_matrix, axis=0))
    print('Std: ', np.std(res_matrix, axis=0))
    f.close()

    path = f"simulate_deterministic_overlap/linear/result_avg.csv"
    f_avg = open(path, 'a+')
    if not os.path.getsize(path): # create the first line for the first time.
        f_avg.write(f"method,n,d,SHD_mean,F1_mean,precison_mean,recall_mean,runtime_mean,SHD_std,F1_std,precison_std,recall_std,runtime_std\n")
    my_mean = np.mean(res_matrix, axis=0)
    my_std = np.std(res_matrix, axis=0)
    f_avg.write(f"{method},{n},{d},{my_mean[0]},{my_mean[1]},{my_mean[2]},{my_mean[3]},{my_mean[4]},{my_std[0]},{my_std[1]},{my_std[2]},{my_std[3]},{my_std[4]}\n")
    f_avg.close()

if __name__=='__main__':
    main()
    # res_matrix = []
    # for i in range(10):
    #     ####### In case, the running for DGES is stuck somewhere: just skip this instance #########
    #     # if i==3:
    #     #     continue 

    #     start = time.time()
    #     linear = True    # linear or nonlinear.
    #     n = 500          # sample size: [100, 250, 500, 1000, 2000].
    #     d = 20            # variable: [6, 8, 10, 12, 14, 20, 30].
    #     method = 'astar'  # methods: ['dges', 'astar', 'ges', 'pc'].
    #     res = run_linear_methods(i, n, d, method) 
    #     # res = run_nonlinear_methods(i, n, d, method) 
    #     end = time.time()
    #     print(f"The time cost is: {end-start}s.")
    #     if res is not None:
    #         res.append(end-start)
    #         res_matrix.append(res)

    # res_matrix = np.array(res_matrix)
    # print(f"SHD | F1 | Precision | Recall | Time.")
    # print("Mean: ", np.mean(res_matrix, axis=0))
    # print('Std: ', np.std(res_matrix, axis=0))
    