from torch_geometric.nn.conv import GCNConv
import torch
import torch.nn.functional as F

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super().__init__()
        
        self.dropout = dropout
        
        self.conv1 = GCNConv(in_channels, 2*out_channels) 
        self.conv_mu = GCNConv(2*out_channels, out_channels) 
        self.conv_logstd = GCNConv(2*out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout)
        x = self.conv1(x, edge_index).relu()
        
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        # print(mu.shape, logstd.shape)
        
        return mu, logstd

