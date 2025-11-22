import torch
import numpy as np
import dgl
from dgl import save_graphs, load_graphs
import random
import json

random.seed(16)
def save_graph():

    adj_drug_protein = np.loadtxt('../data/Luos_DTIdata/mat_drug_protein.txt', dtype=np.float32)
    adj_drug_disease = np.loadtxt('../data/Luos_DTIdata/mat_drug_disease.txt', dtype=np.float32)
    adj_drug_se = np.loadtxt('../data/Luos_DTIdata/mat_drug_se.txt', dtype=np.float32)
    adj_protein_disease = np.loadtxt('../data/Luos_DTIdata/mat_protein_disease.txt', dtype=np.float32)


    # edge_type = "protein_disease"

    nums_drug = len(adj_drug_protein)
    nums_protein = len(adj_drug_protein[0])
    nums_disease = len(adj_drug_disease[0])
    nums_se = len(adj_drug_se[0])
#################################################################################

    embedding_drug = np.random.normal(loc=6.0, scale=3.0, size=(nums_drug, 128))
    embedding_protein = np.random.normal(loc=6.0, scale=3.0, size=(nums_protein, 128))
    embedding_disease = np.random.normal(loc=6.0, scale=3.0, size=(nums_disease, 128))
    embedding_se = np.random.normal(loc=6.0, scale=3.0, size=(nums_se, 128))

    embedding_drug = torch.tensor(embedding_drug,dtype=torch.float32)
    embedding_protein = torch.tensor(embedding_protein,dtype=torch.float32)
    embedding_disease = torch.tensor(embedding_disease,dtype=torch.float32)
    embedding_se = torch.tensor(embedding_se,dtype=torch.float32)

#################################################################################
    link_DPPS = {}
    notLink_DPPS = {}
    link_num = {}
    notlink_num = {}

    link_DPPS["drug_protein"] = []
    link_DPPS["drug_disease"] = []
    link_DPPS["drug_se"] = []
    link_DPPS["protein_disease"] = []

    notLink_DPPS["drug_protein"] = []
    notLink_DPPS["drug_disease"] = []
    notLink_DPPS["drug_se"] = []
    notLink_DPPS["protein_disease"] = []

    link_num["drug_protein"] = 0
    link_num["drug_disease"] = 0
    link_num["drug_se"] = 0
    link_num["protein_disease"] = 0

    notlink_num["drug_protein"] = 0
    notlink_num["drug_disease"] = 0
    notlink_num["drug_se"] = 0
    notlink_num["protein_disease"] = 0

    node_labels = []
    graph_DPPNode = []
    # has_embedding_DPPs = [[0 for i in range(nums_disease)] for _ in range(nums_protein)]

    #   drug-protein
    for i in range(nums_drug):
        for j in range(nums_protein):
            if adj_drug_protein[i][j] == 1:
                link_DPPS["drug_protein"].append((("drug","protein"),(i,j)))
                link_num["drug_protein"] += 1
            else:
                notLink_DPPS["drug_protein"].append((("drug","protein"),(i,j)))
                notlink_num["drug_protein"] += 1

    # drug-disease
    for i in range(nums_drug):
        for j in range(nums_disease):
            if adj_drug_disease[i][j] == 1:
                link_DPPS["drug_disease"].append((("drug","disease"),(i,j)))
                link_num["drug_disease"] += 1
            else:
                notLink_DPPS["drug_disease"].append((("drug","disease"),(i,j)))
                notlink_num["drug_disease"] += 1

    # drug-se
    for i in range(nums_drug):
        for j in range(nums_se):
            if adj_drug_se[i][j] == 1:
                link_DPPS["drug_se"].append((("drug","se"),(i,j)))
                link_num["drug_se"] += 1
            else:
                notLink_DPPS["drug_se"].append((("drug","se"),(i,j)))
                notlink_num["drug_se"] += 1

    # protein-disease
    for i in range(nums_protein):
        for j in range(nums_disease):
            if adj_protein_disease[i][j] == 1:
                link_DPPS["protein_disease"].append((("protein","disease"),(i,j)))
                link_num["protein_disease"] += 1
            else:
                notLink_DPPS["protein_disease"].append((("protein","disease"),(i,j)))
                notlink_num["protein_disease"] += 1

    graph_DPPNode += random.sample(link_DPPS["drug_protein"], 333)
    graph_DPPNode += random.sample(notLink_DPPS["drug_protein"], 333)
    graph_DPPNode += random.sample(link_DPPS["drug_disease"], 333)
    graph_DPPNode += random.sample(notLink_DPPS["drug_disease"], 333)
    graph_DPPNode += random.sample(link_DPPS["drug_se"], 333)
    graph_DPPNode += random.sample(notLink_DPPS["drug_se"], 333)
    graph_DPPNode += random.sample(link_DPPS["protein_disease"], 333)
    graph_DPPNode += random.sample(notLink_DPPS["protein_disease"], 333)


    temp = graph_DPPNode[-1]
    graph_DPPNode[-1] = graph_DPPNode[-2]
    graph_DPPNode[-2] = temp


    node_labels += [[1] for _ in range(333)]
    node_labels += [[0] for _ in range(333)]
    node_labels += [[1] for _ in range(333)]
    node_labels += [[0] for _ in range(333)]
    node_labels += [[1] for _ in range(333)]
    node_labels += [[0] for _ in range(333)]
    node_labels += [[1] for _ in range(333)]
    node_labels += [[0] for _ in range(333)]


    graph_DPPNode = [(i, item) for i, item in enumerate(graph_DPPNode)]

    lis_link_source = []
    lis_link_sink = []
    lis_embedding = []

    for i in graph_DPPNode:
        for j in graph_DPPNode:
            if i[1] != j[1] and (i[1][0][0]+str(i[1][1][0]) == j[1][0][0]+str(j[1][1][0]) or i[1][0][1]+str(i[1][1][1]) == j[1][0][1]+str(j[1][1][1])):
                lis_link_source.append(i[0])
                lis_link_sink.append(j[0])

    for i,j in (graph_DPPNode):
        if j[0][0] == "drug" and j[0][1] == "protein":
            em = torch.cat((embedding_drug[j[1][0]],embedding_protein[j[1][1]]),dim=0)
        elif j[0][0] == "drug" and j[0][1] == "disease":
            em = torch.cat((embedding_drug[j[1][0]], embedding_disease[j[1][1]]), dim=0)
        elif j[0][0] == "drug" and j[0][1] == "se":
            em = torch.cat((embedding_drug[j[1][0]], embedding_se[j[1][1]]), dim=0)
        elif j[0][0] == "protein" and j[0][1] == "disease":
            em = torch.cat((embedding_protein[j[1][0]], embedding_disease[j[1][1]]), dim=0)
        # em = torch.randn(1024)  消融
        lis_embedding.append(em)

    g = dgl.graph((lis_link_source, lis_link_sink))
    node_labels = torch.tensor(node_labels)
    g.ndata['label'] = node_labels
    lis_embedding= torch.tensor([item.cpu().detach().numpy() for item in lis_embedding])
    g.ndata['feature'] = lis_embedding
    save_path = f"../graph_dti/luo_graph_dti/random_all/init_graph_all"
    save_graphs(save_path, [g])

save_graph()