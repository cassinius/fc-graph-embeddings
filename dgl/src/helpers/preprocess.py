import torch
import torch.nn.functional as F
import numpy as np
from ast import literal_eval

import spacy
# Load the spacy model that you have installed
nlp = spacy.load('en_core_web_md')


def string_arr_2_doc_vec(raw_col):
  doc_vecs = [nlp(line).vector for line in raw_col]
  return doc_vecs


def str_arr_col_2_str(raw):
  string_col = list(raw.apply(literal_eval))
  pure_str_col = [" ".join(list(line_arr)) for line_arr in string_col]
  return pure_str_col


def string_arr_col_2_doc_vec(raw):
  pure_str_col = str_arr_col_2_str(raw)
  doc_vecs = [nlp(line_str).vector for line_str in pure_str_col]
  return doc_vecs


def enrich_graph_with_str_features(graph, node, data, feature):
  docvecs = string_arr_2_doc_vec(data[feature])
  graph.nodes[node].data[feature] = torch.tensor(docvecs)
  print(f"{feature} string docvec of node #1:\n", docvecs[0])


def enrich_graph_with_str_arr_features(graph, node, data, feature):
  docvecs = string_arr_col_2_doc_vec(data[feature])
  graph.nodes[node].data[feature] = torch.tensor(docvecs)
  print(f"{feature} string array docvec of node #1:\n", docvecs[0])


def enrich_graph_with_numeric_features(graph, node, data, feature):
  min_val = min(data[feature])
  max_val = max(data[feature])
  features = torch.tensor(data[feature].to_numpy()).float() / (max_val - min_val)
  graph.nodes[node].data[feature] = features
  print(f"{feature} numeric feature of node #1:\n", features[0])


def enrich_graph_with_onehot_features(graph, node, data, feature):
  attr_list = data[feature].to_list()
  unique_attrs = set(attr_list)
  unique_attrs_dict = dict(zip(unique_attrs, np.arange(len(unique_attrs))))
  attr_ids = torch.tensor([unique_attrs_dict[type] for type in attr_list]).long()
  one_hots = F.one_hot(attr_ids)
  graph.nodes[node].data[feature] = one_hots
  print(f"{feature} one-hot feature of node #1:\n", one_hots[0])

