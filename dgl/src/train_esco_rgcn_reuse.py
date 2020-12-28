from pathlib import Path
from train_rgcn import prepare_and_perform_training


esco_file = Path(__file__).parent / "../../trained_models/esco_graph_feats.bin"


# # numeric job features -> to one-hot followed by pca
# job_iscos = embeddings_from_numeric(esco_graph.nodes['job'].data['isco-common'])


# Right now, we're only supporting pre-computed features formatted as tensors of same length
# Todo: add info about how to pre-process the indivdual node attributes
init_embeddings = {
    "skill": {
        "attrs": ["preferredLabel", "altLabels", "description"],
        "aggregator": "sum"
    },
    "job": {
        "attrs": ["preferredLabel", "altLabels", "description"],
        "aggregator": "sum"
    }
}


if __name__ == "__main__":
    target_type = 'skill'
    target_attr = 'reuse'
    prepare_and_perform_training(esco_file, target_type, target_attr, init_embeddings)
