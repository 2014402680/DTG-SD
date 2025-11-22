from dgl import save_graphs, load_graphs

save_path_train = "./tam_graph_dti/Es/graph_all"
graph_train = load_graphs(save_path_train)
save_path_dev = "./tam_graph_dti/Es/init_graph_all"
graph_dev = load_graphs(save_path_dev)


a = 0