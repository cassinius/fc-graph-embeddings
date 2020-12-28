"""
  NOTES

  Since the input data come from our own database where we have tidly
  connected our nodes & all _id's are present, accounted for and valid,
  we can use .map instead of .replace here (even without `fillna`)
"""

from pathlib import Path
import networkx as nx
import pandas as pd
import numpy as np
import torch
import dgl
from dgl.data.utils import save_graphs

import matplotlib.pyplot as plt

from helpers.preprocess import string_arr_col_2_doc_vec

hierarchy_out_file = Path(__file__).parent / "../trained_models/job_hierarchy_graph.bin"

"""
  Input
"""
esco_csv_dir = Path(__file__).parent / '../../data/esco/csvs'

jobs_data = pd.read_csv(esco_csv_dir / 'e_jobs.csv')
broader_occ_data = pd.read_csv(esco_csv_dir / 'e_broaderJob.csv')


"""
  Re-labeling data for node-IDs to be consecutive integers from 0
"""
# We need a dictionary {key: _id, value: <row_number>}

# Row numbers starting at zero
# job_ids = np.arange(len(jobs_data))
# this would also work
job_ids = np.arange(jobs_data.shape[0])
jobs_id_dict = dict(zip(jobs_data['_id'], job_ids))
# print(jobs_id_dict)

# Now replace _id column in `jobs_data` with jobs_ids
# jobs_data['_id'].replace(jobs_id_dict, inplace=True)
# Map works *muchus fasteros*, but doesn't work in place...
jobs_data['_id'] = jobs_data['_id'].map(jobs_id_dict)
# print(jobs_data)

# same with broader_occ -> replace _from & _to
broader_occ_data['_from'] = broader_occ_data['_from'].map(jobs_id_dict)
broader_occ_data['_to'] = broader_occ_data['_to'].map(jobs_id_dict)
# print(broader_occ_data)


""" 
  Construct a simple graph just of occupations and their hierarchy
"""
src = broader_occ_data['_from'].to_numpy()
dst = broader_occ_data['_to'].to_numpy()
g = dgl.graph((src, dst))
print('Job hierarcy graph: ', g)


"""
  Visualize
"""
nxg = g.to_networkx().to_undirected()
# Takes too long...
# pos = nx.kamada_kawai_layout(nx)
# pos = nx.spring_layout(nxg)
# nx.draw(nxg, pos, with_labels=True, node_color=[[.7, .7, .7]])
# plt.show()


"""
  Query
"""
print('#Nodes', g.number_of_nodes())
print('#Edges', g.number_of_edges())
print('In-degree of Node #0', g.in_degrees(0))
print('Out-degree of Node #0', g.out_degrees(0))
print('Successors of node #639', g.successors(639))


"""
 Pre-processing the job features
"""
# first, let's see what Pandas made of different csv input columns
# especially, if it was able to correctly parse array data... NOPE ;-)
# print(jobs_data['preferredLabel'])
# print(jobs_data['altLabels'])

alt_labels = string_arr_col_2_doc_vec(jobs_data['altLabels'])
print('Document vector for node #639\n', alt_labels[639])
g.ndata['alt_labels'] = torch.tensor(alt_labels)


"""
  Save graph
"""
graph_labels = {"glabel": torch.tensor([0])}
save_graphs(str(hierarchy_out_file), [g], graph_labels)
