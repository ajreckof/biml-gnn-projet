from torch_geometric.nn.conv import GCNConv
import torch

class Decoder(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.linear1 = torch.nn.Linear(2*in_channels, in_channels)
        self.linear2 = torch.nn.Linear(in_channels, 1)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x, edge_index, sigmoid=True):
        # print(x.shape)
        x = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        # print(x.shape)
        # print("----")
        if sigmoid:
            return self.sig(x)
        else:
            return x

