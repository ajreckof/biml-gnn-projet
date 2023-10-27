import torch
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import VGAE

import warnings

from encoder import Encoder

from matplotlib import pyplot as plt
import time
import numpy as np

from multiprocessing import Pool
import signal
from tqdm import tqdm
from IPython.display import clear_output

# device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")

def fit(model, data, epochs, verbose=True, test_interval = 100):
	tests = []
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
	model.train()
	for epoch in range(epochs+1):
		if epoch % test_interval == 0 :
			z = model.encode(data.x, data.train_pos_edge_index)
			model.eval()
			tests.append(model.test(z, data.test_pos_edge_index, data.test_neg_edge_index))
			model.train()
			if verbose :
				print(epoch, " : ", tests[-1])
		optimizer.zero_grad()
		z = model.encode(data.x, data.train_pos_edge_index)
		loss = model.recon_loss(z, data.train_pos_edge_index) + (1 / data.num_nodes) * model.kl_loss()
		loss.backward()
		optimizer.step()
	return tests

def loop_func(entries):
	data, dropouts, epochs, file_path, test_interval = entries
	AUCs = {}
	APs = {}
	dt = data.__copy__().to(device)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		train_test_split_edges(dt)
	dt.num_classes = len(set(data.country))
	for dropout in dropouts:
		print("Dropout : ", dropout)
		
		encoder = Encoder(in_channels=dt.num_features, out_channels=15, dropout=dropout)
		vgae = VGAE(encoder).to(device)
		
		start_time = time.time()
		AUC, AP = zip(*fit(vgae, dt, epochs, verbose=False, test_interval= test_interval))
		AUCs[dropout] = AUC
		APs[dropout] = AP
		elapsed_time = time.time() - start_time

		vgae.eval()
		z = vgae.encode(dt.x, dt.train_pos_edge_index)
		AUC, AP = vgae.test(z, dt.test_pos_edge_index, dt.test_neg_edge_index)
		
		with open(file_path, "a") as f:
			f.write(f"{dropout};{AUC};{AP};{epochs};{elapsed_time}\n")
		print(f"{dropout};{AUC};{AP};{epochs};{elapsed_time}")
	return AUCs, APs

def test_dropout(file_path, data, dropouts, epochs, n = 10, test_interval= 10, average_on = 10, pool_worker = 4):

	try :
		open(file_path, "r")
	except FileNotFoundError as e:
		with open(file_path, "w") as f:
			f.write("Dropout;AUC;AP;epochs;elapsed_time;\n")

	AUCs = { dropout : [] for dropout in dropouts}
	APs = { dropout : [] for dropout in dropouts}

	pool = Pool(pool_worker)

	def handle_interrupt(signal, frame):
		pool.terminate()  # Terminate the pool of worker processes
		pool.join()  # Wait for the pool to clean up
		print("Main process interrupted. Worker processes terminated.")
		exit(1)

	# Register the signal handler for interrupt signals
	signal.signal(signal.SIGINT, handle_interrupt)
	

	for AUC, AP in tqdm(pool.imap_unordered(loop_func,[(data, dropouts, epochs, file_path, test_interval)] * n)): 
		for dropout in dropouts :
			AUCs[dropout].append(AUC[dropout])
			APs[dropout].append(AP[dropout])




	clear_output(wait=True)
	draw_AUCs_APs(AUCs,APs, average_on, test_interval)
	return AUCs, APs


def draw_AUCs_APs(AUCs,APs, average_on, test_interval):
	mean_AUC, mean_AP = {}, {}
	std_AUC, std_AP = {}, {}
	
	def floating_average(list):
		return [ sum(list[max(0,i-average_on):i+1])/ min(i+1,average_on) for i in range(len(list))]

	for dropout in AUCs :
		mean_AUC[dropout] = floating_average([ np.mean(x)  for x in zip(*AUCs[dropout])])
		mean_AP[dropout] = floating_average([ np.mean(x)  for x in zip(*APs[dropout])])
		std_AUC[dropout] = floating_average([ np.std(x) for x in zip(*AUCs[dropout])])
		std_AP[dropout] = floating_average([ np.std(x)  for x in zip(*APs[dropout])])
	fig = plt.figure(figsize = (20,20))
	ax1 = fig.add_subplot(2,2,1)
	ax1.grid(visible= True, which='both')
	ax1.set_xlabel("epoch")
	ax1.set_ylabel("AUC")

	ax2 = fig.add_subplot(2,2,2)
	ax2.grid(visible= True, which='both')
	ax2.set_xlabel("epoch")
	ax2.set_ylabel("AP")
	
	ax3 = fig.add_subplot(2,2,3)
	ax3.grid(visible= True, which='both')
	ax3.set_xlabel("epoch")
	ax3.set_ylabel("AUC")

	ax4 = fig.add_subplot(2,2,4)
	ax4.grid(visible= True, which='both')
	ax4.set_xlabel("epoch")
	ax4.set_ylabel("AP")

	for dropout in AUCs :
		ax1.plot(np.arange(len(mean_AUC[dropout])) * test_interval, mean_AUC[dropout], label = dropout)
		ax2.plot(np.arange(len(mean_AP[dropout])) * test_interval, mean_AP[dropout], label = dropout)
		ax3.plot(np.arange(len(std_AUC[dropout])) * test_interval, std_AUC[dropout], label = dropout)
		ax4.plot(np.arange(len(std_AP[dropout])) * test_interval, std_AP[dropout], label = dropout)
	
	ax1.legend(loc='upper left')
	ax2.legend(loc='upper left')    
	ax3.legend(loc='upper left')
	ax4.legend(loc='upper left')
	print("start plotting")
	plt.show()
