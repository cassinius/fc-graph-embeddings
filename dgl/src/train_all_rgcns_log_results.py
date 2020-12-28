import csv
import json
import configparser
from pathlib import Path

from train_rgcn import prepare_and_perform_training


graph_file_dir = Path(__file__).parent / "../../trained_models"
experiments_file = Path(__file__).parent / "./config/rgcn_experiments.json"
config_file = Path(__file__).parent / "./config/rgcn.config"

config = configparser.ConfigParser()
config.read(config_file)
lr = float(config['DEFAULT']['lr'])
epochs = int(config['DEFAULT']['epochs'])
in_dim = int(config['DEFAULT']['in_dim'])
hidden_dim = int(config['DEFAULT']['hidden_dim'])
train_split = float(config['DEFAULT']['train_split'])
val_split = float(config['DEFAULT']['val_split'])

res_file = Path(__file__).parent / \
    f"../../results/rgcn_outputs_lr_{lr}_e_{epochs}_in_{in_dim}_h_{hidden_dim}_ts_{train_split}_vs_{val_split}.csv"


with open(str(experiments_file), 'r') as file:
    experiments = json.loads(file.read())['experiments']


with open(res_file, mode='w') as csv_file:
    fieldnames = ['name', 'graph', 'target_type', 'target_attr',
                  'out_dim', 'best_train', 'best_val', 'best_test']
    # print("Fieldnames: ", fieldnames)
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for experiment in experiments:
        graph = experiment['graph']
        target_type = experiment['target_type']
        target_attr = experiment['target_attr']
        init_embed_config = experiment.get('init_embed_config', None)

        print("-------------------------------------------------------------------")
        print(
            f"\nRunning supervised RGCN training on graph={graph}, target_type={target_type}, target_attr={target_attr}\n")
        print("-------------------------------------------------------------------")

        lr, epochs, in_dim, hidden_dim, out_dim, best_train, best_val, best_test = prepare_and_perform_training(
            graph_file_dir / experiment['graph_file'],
            target_type,
            target_attr,
            init_embed_config
        )

        writer.writerow({
            'name': experiment['name'],
            'graph': graph,
            'target_type': target_type,
            'target_attr': target_attr,
            'out_dim': out_dim,
            'best_train': float(best_train),
            'best_val': float(best_val),
            'best_test': float(best_test)
        })

