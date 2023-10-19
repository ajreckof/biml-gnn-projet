from torch_geometric.nn.conv import GCNConv
import torch

class Decoder(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.conv1 = GCNConv(in_channels, in_channels)
        self.linear1 = torch.nn.Linear(in_channels, in_channels)
        self.linear2 = torch.nn.Linear(in_channels, 1)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x, edge_index, sigmoid=True):
        # print(x.shape)
        x = self.conv1(x, edge_index).relu()
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        # print(x.shape)
        # print("----")
        if sigmoid:
            return self.sig(x)
        else:
            return x

