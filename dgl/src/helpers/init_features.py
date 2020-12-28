from typing import Optional, Sequence
import numpy as np
import torch
from sklearn.decomposition import PCA


def aggregate_embeddings(tensors: Sequence[torch.Tensor], aggregator: str):
  """
    Pre-conditions
    - must be of same length -> torch.stack takes care of that
  """
  if aggregator == 'mean':
    return torch.mean(torch.stack(tensors), dim=0)

  if aggregator == 'sum':
    return torch.sum(torch.stack(tensors), dim=0)
  
  if aggregator == 'min':
    vals, _ = torch.min(torch.stack(tensors), dim=0)
    return vals

  if aggregator == 'max':
    vals, _ = torch.max(torch.stack(tensors), dim=0)
    return vals


def embeddings_from_numeric(numeric_tensor: torch.Tensor):
  codes = numeric_tensor.tolist()
  # print(f"Numeric codes: {codes}, length: {len(codes)}")
  unique_codes = set(codes)
  # print(f"Unique codes: {unique_codes}, length: {len(unique_codes)}")
  codes_dict = dict(zip(unique_codes, np.arange(len(unique_codes))))
  # print("Codes dict:\n", codes_dict)
  codes_normalized = [codes_dict.get(code) for code in codes]
  # print("Normalized codes:\n", codes_normalized)
  codes_onehot = [np.zeros(len(codes)) for i in codes_normalized]
  for i in range(len(codes_onehot) - 1):
    codes_onehot[i][codes_normalized[i]] = 1.0
  pca = PCA(n_components=300)
  pca.fit(codes_onehot)
  return torch.transpose(torch.tensor(pca.components_), 0, 1).long()

