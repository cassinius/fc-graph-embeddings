import configparser

from pathlib import Path
from dgl.data.utils import load_graphs
import numpy as np
import torch

from models.hetero_rgcn import train_and_eval_rgcn, RGCNArgs
from helpers.init_features import aggregate_embeddings, embeddings_from_numeric


config_file = Path(__file__).parent / "./config/rgcn.config"
config = configparser.ConfigParser()
config.read(config_file)
lr = float(config['DEFAULT']['lr'])
epochs = int(config['DEFAULT']['epochs'])
in_dim = int(config['DEFAULT']['in_dim'])
hidden_dim = int(config['DEFAULT']['hidden_dim'])
train_split = float(config['DEFAULT']['train_split'])
val_split = float(config['DEFAULT']['val_split'])


def calculate_init_embeddings(G, init_embed_dict):
    """ Compute pre-initialized node features

    Input types and their transformation:
        * tensors -> aggreate(mean, sum, max)
        * numeric -> transform to one-hot, then PCA -> tensors
        * one-hot -> not defined yet (usually lower-dim than (w2v) tensors)

    Todo:
        * type this stuff !
    """
    init_embeddings = {}
    for node, node_embed_dict in init_embed_dict.items():
        print(node, node_embed_dict)
        pre_computed_tensors = []
        for attr in node_embed_dict['attrs']:
            pre_computed_tensors.append(G.nodes[node].data[attr])
        init_embeddings[node] = aggregate_embeddings(pre_computed_tensors, node_embed_dict['aggregator'])

    return init_embeddings


def prepare_and_perform_training(graph_file, target_type: str, target_attr: str, init_embed_dict=None):
    """ Train supervised on `target_type`->`target_attr` """
    graph_list, _ = load_graphs(str(graph_file))
    G = graph_list[0]
    print(G)

    init_embeddings = calculate_init_embeddings(G, init_embed_dict) if init_embed_dict else None

    labels = torch.tensor(G.nodes[target_type].data[target_attr]).long()
    unique_types = torch.unique(labels)

    nr_targets = G.num_nodes(target_type)
    target_ids = np.arange(nr_targets)
    shuffle = np.random.permutation(target_ids)

    t_up = int(train_split * nr_targets)
    v_up = int(val_split * nr_targets)
    train_idx = torch.tensor(shuffle[0:t_up]).long()
    val_idx = torch.tensor(shuffle[t_up:v_up]).long()
    test_idx = torch.tensor(shuffle[v_up:]).long()

    best_train, best_val, best_test = train_and_eval_rgcn(G, RGCNArgs(
        epochs,
        in_dim,
        hidden_dim,
        len(unique_types),
        target_type,
        train_idx,
        val_idx,
        test_idx,
        labels,
        init_embeddings
    ))

    return lr, epochs, in_dim, hidden_dim, len(unique_types), best_train, best_val, best_test
