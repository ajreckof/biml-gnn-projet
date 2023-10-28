import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_out)
  
    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        return x