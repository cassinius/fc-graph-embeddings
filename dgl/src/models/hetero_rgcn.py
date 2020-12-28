from typing import Optional, Sequence
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCNArgs:
    """
    Using class as a pendant to a Typescript interface

    TODO cannot believe that Python actually needs that whole explicit constructor thing... that's stone age
    """
    epochs: int
    in_dim: int
    hidden_dim: int
    out_dim: int
    out_type: str
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    labels: torch.Tensor
    # TODO replace with something better...
    init_embeddings: object

    def __init__(self,
                 epochs: int,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 out_type: str,
                 train_idx: torch.Tensor,
                 val_idx: torch.Tensor,
                 test_idx: torch.Tensor,
                 labels: torch.Tensor,
                 init_embeddings) -> None:
        self.epochs = epochs
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.out_type = out_type
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.labels = labels
        self.init_embeddings = init_embeddings


class HeteroRGCNLayer(nn.Module):
    """ A single RGCN Layer """

    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        """all edge types get a Linear model ??"""
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in etypes
        })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each types
        funcs = {}

        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean(
                'm', 'h'))  # <- Aggregator (for each rel.)
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        """ This one performs the actual update """
        G.multi_update_all(
            funcs, 'sum')  # <- go through all the relation-wise funcs, then reduce
        # return the updated node feature dictionary
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}


class HeteroRGCN(nn.Module):
    """ The whole GNN """

    def __init__(self, G, in_size, hidden_size, out_size, init_embeddings):
        super(HeteroRGCN, self).__init__()

        """
            Use trainable node embeddings as featureless inputs.
            yiels tensor matrix of #nodes * in_size
        """
        embed_dict = {ntype: nn.Parameter(torch.Tensor(
            G.number_of_nodes(ntype), in_size)) for ntype in G.ntypes}
        """
            initializing weights
            TODO replace with actual (pre-processed) node features vectors
        """
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)

        """
            HACK: override randomly initialized weights by provided embeddings
        """
        if init_embeddings:
            for key, embeddings in init_embeddings.items():
                print(f"\nRGCN - Initializing {key} embeddings with \n{embeddings}")
                embed_dict[key] = nn.Parameter(embeddings)

        self.embed = nn.ParameterDict(embed_dict)
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)

    def forward(self, G, ret_obj):
        h_dict = self.layer1(G, self.embed)
        h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        # return logits for desirec objects (e.g. 'jobs', 'skills', 'products', ...)
        return h_dict[ret_obj]


def train_and_eval_rgcn(G, args: RGCNArgs):
    """ The actual training function """
    model = HeteroRGCN(G, args.in_dim, args.hidden_dim, args.out_dim, args.init_embeddings)
    # print(model)

    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    """ We are passing the whole model to the optimizer, which will handle its internals """

    best_train_acc = 0.0
    best_val_acc = 0.0
    best_test_acc = 0.0

    train_idx = args.train_idx
    val_idx = args.val_idx
    test_idx = args.test_idx
    labels = args.labels

    for epoch in range(args.epochs):
        logits = model(G, args.out_type)

        # print("Logits:", logits)
        # print("Logits[train_idx]:", logits[train_idx])
        # print("Labels[train_idx]:", labels[train_idx])
        # print("Length of Logits[train_idx]:", len(logits[train_idx]))
        # print("Length of Labels[train_idx]:", len(labels[train_idx]))

        # Computing the loss only for labeled nodes
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])

        """
        argmax takes the indices which maximize a probability
        - parameter (1):
        """
        pred = logits.argmax(1)

        train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
        val_acc = (pred[val_idx] == labels[val_idx]).float().mean()
        test_acc = (pred[test_idx] == labels[test_idx]).float().mean()

        if best_train_acc < train_acc:
            best_train_acc = train_acc
        if best_val_acc < val_acc:
            best_val_acc = val_acc
        if best_test_acc < test_acc:
            best_test_acc = test_acc

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 5 == 0:
            print('Epoch: %.2d, Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
                epoch,
                loss,
                train_acc,
                val_acc,
                best_val_acc,
                test_acc,
                best_test_acc,
            ))

    return best_train_acc, best_val_acc, best_test_acc
