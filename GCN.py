import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
	"""Graph Convolutional Network"""
	def __init__(self, dim_in, dim_out, dim_h):
		super().__init__()
		self.gcn1 = GCNConv(dim_in, dim_h)
		self.gcn2 = GCNConv(dim_h, dim_out)
  
	def forward(self, x, edge_index):
		x = self.gcn1(x, edge_index)
		x = torch.relu(x)
		x = self.gcn2(x, edge_index)
		return F.log_softmax(x, dim=1)

	def fit(self, data, epochs):
		criterion = torch.nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
		self.train()
		for epoch in range(epochs+1):
			optimizer.zero_grad()
			out = self(data.x, data.edge_index)
			loss = criterion(out[data.train_mask],
			data.y[data.train_mask])
			loss.backward()
			optimizer.step()
   
class SimpleGCN(torch.nn.Module):
	"""Graph Convolutional Network"""
	def __init__(self, dim_in, dim_out):
		super().__init__()
		self.gcn = GCNConv(dim_in, dim_out)
  
	def forward(self, x, edge_index):
		x = self.gcn(x, edge_index)
		return F.log_softmax(x, dim=1)

	def fit(self, data, epochs):
		criterion = torch.nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
		self.train()
		for epoch in range(epochs+1):
			optimizer.zero_grad()
			out = self(data.x, data.edge_index)
			loss = criterion(out[data.train_mask],
			data.y[data.train_mask])
			loss.backward()
			optimizer.step()