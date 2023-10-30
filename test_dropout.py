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


def draw_AUCs_or_APs(AUCs, average_on, test_interval, values = None, revlog = False, percentiles_base = [0.1, 1, 5, 10, 25, 33, 40,45], output = "output"):
	if values == None :
		values = list(AUCs.keys())
	mean = {}
	median = {}
	std = {}
	percentiles_dict = {}
	
	def floating_average(list):
		return [ sum(list[max(0,i+1-average_on):i+1])/ min(i+1,average_on) for i in range(len(list))]
	
	percentiles = percentiles_base + [ 100 - x for x in reversed(percentiles_base)]

	if revlog:
		f= lambda x : 1-x
	else : 
		f= lambda x: x

	for dropout in values :
		mean[dropout] = floating_average([ f(np.mean(x))  for x in zip(*AUCs[dropout])])
		std[dropout] = floating_average([ np.std(x) for x in zip(*AUCs[dropout])])
		percentiles_dict[dropout] = [floating_average([f(y) for y in x]) for x in zip(*[np.percentile(x, percentiles) for x in zip(*AUCs[dropout])])]
		median[dropout] = floating_average([ f(np.median(x))  for x in zip(*AUCs[dropout])])

	val = np.linspace(0, 1, len(AUCs))
	cmap = plt.get_cmap('rainbow')  
	color_list = [cmap(value) for value in val]
	colors = {dropout : color for dropout,color in zip(AUCs.keys(),color_list)}
	
	fig = plt.figure(figsize = (15,22))
	ax = fig.add_subplot(3,2,1)
	ax.grid(visible= True, which='both')
	ax.set_xlabel("epoch")
	ax.set_ylabel("moyenne")
	for dropout in values :
		ax.plot(np.arange(len(mean[dropout])) * test_interval, mean[dropout], label = dropout, color = colors[dropout])
	if revlog :
		ax.set_yscale('log')
	ax.legend(loc='upper left')
	plt.savefig(f"output/{output}_mean.svg.pdf", bbox_inches='tight')

	fig = plt.figure(figsize = (15,22))
	ax = fig.add_subplot(3,2,1)
	ax.grid(visible= True, which='both')
	ax.set_xlabel("epoch")
	ax.set_ylabel("écart-type")
	for dropout in values :
		ax.plot(np.arange(len(std[dropout])) * test_interval, std[dropout], label = dropout, color = colors[dropout])
	if revlog :
		ax.set_yscale('log')
	ax.legend(loc='upper left')
	plt.savefig(f"output/{output}_std.svg.pdf", bbox_inches='tight')
	
	
	fig = plt.figure(figsize = (15,22))
	ax = fig.add_subplot(3,2,1)
	ax.grid(visible= True, which='both')
	ax.set_xlabel("epoch")
	ax.set_ylabel("médiane")
	for dropout in values :
		ax.plot(np.arange(len(median[dropout])) * test_interval, median[dropout], label = dropout, color = colors[dropout])
		for i in range(len(percentiles_base)) : 
			ax.fill_between(np.arange(len(median[dropout])) * test_interval, percentiles_dict[dropout][i], percentiles_dict[dropout][-i-1], color = colors[dropout], alpha = 0.1)
	if revlog :
		ax.set_yscale('log')
	ax.legend(loc='upper left')
	plt.savefig(f"output/{output}_cinf.svg.pdf", bbox_inches='tight')

	plt.show()
