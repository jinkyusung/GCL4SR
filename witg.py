import torch
import numpy as np
from torch_geometric.data import Data



def read_raw_data(file):
    lines = open(file).readlines()
    user_sequences = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(',')
        items = [int(item) for item in items]
        user_sequences.append(items)
        item_set = item_set | set(items)

    num_node = max(item_set) + 1
    return user_sequences, num_node



def convert_pyg_data(adj, num_node):
    adj_pyg = []
    weight_pyg = []

    for t in range(1, num_node):
        x = [v for v in sorted(adj[t].items(), reverse=True, key=lambda x: x[1])]
        adj_pyg += [[t, v[0]] for v in x]
        weight_pyg += [v[1] for v in x]

    adj_np = np.array(adj_pyg)
    adj_np = adj_np.transpose()
    edge_np = np.array([adj_np[0, :], adj_np[1, :]])
    x = torch.arange(0, num_node).long().view(-1, 1)
    edge_attr = torch.from_numpy(np.array(weight_pyg)).view(-1, 1)
    edge_index = torch.from_numpy(edge_np).long()
    graph = Data(x, edge_index, edge_attr=edge_attr)
    return graph



def build_weighted_item_transition_graph(train_sequence_file):
    user_sequences, num_node = read_raw_data(train_sequence_file)
    adj = [dict() for _ in range(num_node)]
    # ======================= TODO ========================= #
    relation = []
    for i in range(len(user_sequences)):
        data = user_sequences[i]
        for k in range(1, 4):
            for j in range(len(data) - k):
                relation.append([data[j], data[j + k], k])
                relation.append([data[j + k], data[j], k])

    for temp in relation:
        if temp[1] in adj[temp[0]].keys():
            adj[temp[0]][temp[1]] += 1 / temp[2]
        else:
            adj[temp[0]][temp[1]] = 1 / temp[2]
    # ====================================================== #
    graph: Data = convert_pyg_data(adj, num_node)
    return graph
