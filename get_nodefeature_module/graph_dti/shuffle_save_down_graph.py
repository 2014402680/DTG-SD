import os
import random

import torch
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
import numpy as np
from dgl import save_graphs, load_graphs

#########################################################################
save_path = "./luo_graph_dti/n(6,3)_all_link_random/init_graph_all"
save_path_train = "./luo_graph_dti/n(6,3)_all_link_random/graph_train"
save_path_dev = "./luo_graph_dti/n(6,3)_all_link_random/graph_dev"
save_path_all = "./luo_graph_dti/n(6,3)_all_link_random/graph_all"
########################################################################

graph = load_graphs(save_path)

np.random.seed(0)

node_numbers = graph[0][0].number_of_nodes()
# random_node_list = random.sample([i for i in range(666)],666) + random.sample([i for i in range(666,1332)],666) + random.sample([i for i in range(1332,1998)],666) + random.sample([i for i in range(1998,2664)],666)
node_list = [i for i in range(123200)]
piece = int(node_numbers/12)
train_graph = []
dev_graph = []
all_graph = []

for i in range(10):
    train_graph.append(graph[0][0].subgraph(node_list[i*piece:(i+1)*piece]))
dev_graph.append(graph[0][0].subgraph(node_list[piece*10:piece*11]))
dev_graph.append(graph[0][0].subgraph(node_list[piece*11:]))


all_graph.append(graph[0][0].subgraph(node_list[:]))

'''
for g in train_graph:
    sample = torch.tensor([[_,_,0] for _ in range(g.number_of_nodes())])
    adj = g.adjacency_matrix(transpose=False)
    for i in range(g.number_of_nodes()):
        if adj[i]._indices().shape[1] == 0:
            continue
        else:
            for j in range(g.number_of_nodes()):
                if j not in adj[i]._indices():
                    sample[i] = torch.Tensor([i,adj[i]._indices()[0][0],j])
                    break
    g.ndata["sample"] = sample

for g in dev_graph:
    sample = torch.tensor([[_,_,0] for _ in range(g.number_of_nodes())])
    adj = g.adjacency_matrix(transpose=False)
    for i in range(g.number_of_nodes()):
        if adj[i]._indices().shape[1] == 0:
            continue
        else:
            for j in range(g.number_of_nodes()):
                if j not in adj[i]._indices():
                    sample[i] = torch.Tensor([i,adj[i]._indices()[0][0],j])
                    break
    g.ndata["sample"] = sample

'''

for g in all_graph:
    random_list = [i for i in range(g.number_of_nodes())]
    sample = torch.tensor([[_,_,_] for _ in range(g.number_of_nodes())])
    adj = g.adjacency_matrix(transpose=False)
    for i in range(g.number_of_nodes()):
        if i % 100 == 0:
            print(i)
        random.shuffle(random_list)
        if adj[i]._indices().shape[1] == 0:
            for j in random_list:
                sample[i] = torch.Tensor([i, i, j])
                break
        else:
            for j in random_list:
                if j not in adj[i]._indices():
                    sample[i] = torch.Tensor([i,adj[i]._indices()[0][0],j])
                    break
    g.ndata["sample"] = sample



save_graphs(save_path_train, train_graph)
save_graphs(save_path_dev,dev_graph)
save_graphs(save_path_all,all_graph)