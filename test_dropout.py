import torch
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import VGAE, GAE
from GCN import GCN

import warnings

from encoder import Encoder
from decoder import Decoder


from matplotlib import pyplot as plt
import time
import numpy as np

from multiprocessing import Pool
import signal
from tqdm import tqdm
from IPython.display import clear_output

AUCs,APs = {},{}

# device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print("reimported test_dropout")

def set_AUCs_APs(AUC_value,AP_value):
	for x in AUC_value :
		AUCs[x] = AUC_value[x]
		APs[x] = AP_value[x]

def get_AUCs_APs():
	return AUCs,APs

def get_name_from_tuple(t):

	is_vgae_dropout, decoder, encoder_out = t
	is_vgae, dropout = is_vgae_dropout
	output = "VGAE" if is_vgae else "GAE"
	if decoder and dropout :
		output += "_with_decoder_and_dropout"
	elif decoder:
		output += "_with_decoder"
	elif dropout:
		output += "_with_dropout" 
	return output

def fit(model, data, epochs, verbose=True, test_interval = 100):
	tests = []
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
	model.train()
	for epoch in range(epochs+1):
		if epoch % test_interval == 0 :
			with torch.no_grad():
				model.eval()
				z = model.encode(data.x, data.train_pos_edge_index)
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

def fit_gae(model, data, epochs, verbose=True, test_interval = 100):
	tests = []
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
	model.train()
	for epoch in range(epochs+1):
		if epoch % test_interval == 0 :
			with torch.no_grad():
				model.eval()
				z = model.encode(data.x, data.train_pos_edge_index)
				
				tests.append(model.test(z, data.test_pos_edge_index, data.test_neg_edge_index))
				model.train()
		optimizer.zero_grad()
		z = model.encode(data.x, data.train_pos_edge_index)
		loss = model.recon_loss(z, data.train_pos_edge_index)
		loss.backward()
		optimizer.step()
	return tests

def loop_func(entries):
	data, models, epochs, file_path, test_interval = entries
	AUCs = {}
	APs = {}
	transform = RandomLinkSplit(is_undirected=True, split_labels=True, num_val=0)
	train_data, _, test_data = transform(data)
	dt = data.__copy__()
	dt.train_pos_edge_index = train_data.pos_edge_label_index
	dt.test_pos_edge_index = test_data.pos_edge_label_index
	dt.test_neg_edge_index = test_data.neg_edge_label_index

	dt.num_classes = len(set(data.country))
	for model_val in models:
		is_vgae_dropout, decoder, encoder_out = model_val
		is_vgae, dropout = is_vgae_dropout
		if is_vgae :
			model = VGAE(Encoder(dt.num_features, encoder_out, dropout= dropout), Decoder(encoder_out) if decoder else None)
		else :
			model = GAE(GCN(dt.num_features, encoder_out), Decoder(encoder_out) if decoder else None)
		model.name = get_name_from_tuple(model_val)
		
		start_time = time.time()
		if is_vgae :
			AUC, AP = zip(*fit(model, dt, epochs, verbose=False, test_interval= test_interval))
		else : 
			AUC, AP = zip(*fit_gae(model, dt, epochs, verbose=False, test_interval= test_interval))

		AUCs[model.name] = AUC
		APs[model.name] = AP
		elapsed_time = time.time() - start_time

		model.eval()
		z = model.encode(dt.x, dt.train_pos_edge_index)
		AUC, AP = model.test(z, dt.test_pos_edge_index, dt.test_neg_edge_index)
		
		with open(file_path, "a") as f:
			f.write(f"{model.name};{AUC};{AP};{epochs};{elapsed_time}\n")
		print(f"{model.name};{AUC};{AP};{epochs};{elapsed_time}")
	return AUCs, APs

def test_dropout(file_path, data, models, epochs, n = 10, test_interval= 10, average_on = 10, pool_worker = 4, show_every= 10):
	try :
		open(file_path, "r")
	except FileNotFoundError as e:
		with open(file_path, "w") as f:
			f.write("Dropout;AUC;AP;epochs;elapsed_time;\n")

	AUCs = { get_name_from_tuple(model_val) : [] for model_val in models}
	APs = { get_name_from_tuple(model_val) : [] for model_val in models}

	pool = Pool(pool_worker)

	def handle_interrupt(signal, frame):
		pool.terminate()  # Terminate the pool of worker processes
		pool.join()  # Wait for the pool to clean up
		print("Main process interrupted. Worker processes terminated.")
		set_AUCs_APs(AUCs,APs)
		exit(1)

	# Register the signal handler for interrupt signals
	signal.signal(signal.SIGINT, handle_interrupt)
	
	i = 0
	for AUC, AP in tqdm(pool.imap_unordered(loop_func,[(data, models, epochs, file_path, test_interval)] * n)): 
		i+=1
		for model_val in models :
			name = get_name_from_tuple(model_val)
			AUCs[name].append(AUC[name])
			APs[name].append(AP[name])

		if i % show_every == 0 : 
			clear_output(wait= True)
			draw_AUCs_APs(AUCs, APs, average_on, test_interval)




	# clear_output(wait=True)
	print(AUCs, APs)
	draw_AUCs_APs(AUCs,APs, average_on, test_interval)
	return AUCs, APs


def draw_AUCs_APs(AUCs,APs, average_on, test_interval, values = None):
	if values == None :
		values = list(AUCs.keys())
	mean_AUC, mean_AP = {}, {}
	median_AUC, median_AP = {}, {}
	std_AUC, std_AP = {}, {}
	percentiles_AUC, percentiles_AP = {}, {}
	
	def floating_average(list):
		return [ sum(list[max(0,i+1-average_on):i+1])/ min(i+1,average_on) for i in range(len(list))]
	
	percentiles_base = [5, 10, 25, 33, 40,45]
	percentiles = percentiles_base + [ 100 - x for x in reversed(percentiles_base)]



	for dropout in values :
		mean_AUC[dropout] = floating_average([ np.mean(x)  for x in zip(*AUCs[dropout])])
		mean_AP[dropout] = floating_average([ np.mean(x)  for x in zip(*APs[dropout])])
		std_AUC[dropout] = floating_average([ np.std(x) for x in zip(*AUCs[dropout])])
		std_AP[dropout] = floating_average([ np.std(x)  for x in zip(*APs[dropout])])
		percentiles_AUC[dropout] = [floating_average(x) for x in zip(*[np.percentile(x, percentiles) for x in zip(*AUCs[dropout])])]
		percentiles_AP[dropout] = [floating_average(x) for x in zip(*[np.percentile(x, percentiles) for x in zip(*APs[dropout])])]
		median_AUC[dropout] = floating_average([ np.median(x)  for x in zip(*AUCs[dropout])])
		median_AP[dropout] = floating_average([ np.median(x)  for x in zip(*APs[dropout])])

	
	
	fig = plt.figure(figsize = (15,22))
	ax1 = fig.add_subplot(3,2,1)
	ax1.grid(visible= True, which='both')
	ax1.set_xlabel("epoch")
	ax1.set_ylabel("AUC")

	ax2 = fig.add_subplot(3,2,2)
	ax2.grid(visible= True, which='both')
	ax2.set_xlabel("epoch")
	ax2.set_ylabel("AP")
	
	ax3 = fig.add_subplot(3,2,3)
	ax3.grid(visible= True, which='both')
	ax3.set_xlabel("epoch")
	ax3.set_ylabel("AUC")

	ax4 = fig.add_subplot(3,2,4)
	ax4.grid(visible= True, which='both')
	ax4.set_xlabel("epoch")
	ax4.set_ylabel("AP")

	ax5 = fig.add_subplot(3,2,5)
	ax5.grid(visible= True, which='both')
	ax5.set_xlabel("epoch")
	ax5.set_ylabel("AUC")

	ax6 = fig.add_subplot(3,2,6)
	ax6.grid(visible= True, which='both')
	ax6.set_xlabel("epoch")
	ax6.set_ylabel("AP")

	val = np.linspace(0, 1, len(AUCs))
	cmap = plt.get_cmap('rainbow')  
	color_list = [cmap(value) for value in val]
	colors = {dropout : color for dropout,color in zip(AUCs.keys(),color_list)}

	for dropout in values :
		ax1.plot(np.arange(len(mean_AUC[dropout])) * test_interval, mean_AUC[dropout], label = dropout, color = colors[dropout])
		ax2.plot(np.arange(len(mean_AP[dropout])) * test_interval, mean_AP[dropout], label = dropout, color = colors[dropout])
		ax3.plot(np.arange(len(std_AUC[dropout])) * test_interval, std_AUC[dropout], label = dropout, color = colors[dropout])
		ax4.plot(np.arange(len(std_AP[dropout])) * test_interval, std_AP[dropout], label = dropout, color = colors[dropout])
		ax5.plot(np.arange(len(median_AUC[dropout])) * test_interval, median_AUC[dropout], label = dropout, color = colors[dropout])
		ax6.plot(np.arange(len(median_AP[dropout])) * test_interval, median_AP[dropout], label = dropout, color = colors[dropout])
		for i in range(len(percentiles_base)) : 
			ax5.fill_between(np.arange(len(mean_AUC[dropout])) * test_interval, percentiles_AUC[dropout][i], percentiles_AUC[dropout][-i-1], color = colors[dropout], alpha = 0.1)
			ax6.fill_between(np.arange(len(mean_AP[dropout])) * test_interval, percentiles_AP[dropout][i], percentiles_AP[dropout][-i-1], color = colors[dropout], alpha = 0.1)



	
	ax1.legend(loc='upper left')
	ax2.legend(loc='upper left')    
	ax3.legend(loc='upper left')
	ax4.legend(loc='upper left')
	ax5.legend(loc='upper left')
	ax6.legend(loc='upper left')
	print("start plotting")
	plt.savefig("output.svg.pdf")
	plt.show()
