import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

#list containing all files from michele metric
file_list=['iou0.5_score0.5.json', 'iou0.5_score0.75.json', 'iou0.75_score0.5.json', 'iou0.75_score0.75.json']

for file_path in file_list:
	# first i remove the json extension
	base_path, _ = os.path.splitext(file_path)
	with open(file_path, 'r') as json_file:
    		data = json.load(json_file)
	
	#classic font plotlib settings
	font = {#'family' : 'arial',
	        #'weight' : 'bold',
	        'size'   : 22}
	
	plt.rc('font', **font)
	
	#getting data, calculating percentage of tp, fp and fpiou over total_detections
	epochs = [entry['epoch'] for entry in data]
	total_detections = [entry['total_detections'] for entry in data]
	TP_percentage = [(entry['TP'] / entry['total_detections']) for entry in data]
	#FP_percentage = [(entry['FP'] / entry['total_detections']) for entry in data]
	#gli FP non ha senso printarli, perché le pred della faster sono sempre con label 1 di persona, non potrà mai fare una pred con label di background!
	TPm_percentage = [(entry['TPm'] / entry['total_detections']) for entry in data]
	FPiou_percentage = [(entry['FPiou'] / entry['total_detections']) for entry in data]
	
	#plotting now the curves
	plt.figure(figsize=(15, 10))
	plt.plot(epochs, TP_percentage, label='TP Percentage', color='blue')
	#plt.plot(epochs, FP_percentage, label='FP Percentage', color='red')
	plt.plot(epochs, TPm_percentage, label='TPm Percentage', color='green')
	plt.plot(epochs, FPiou_percentage, label='FPiou Percentage', color='red')
	
	#label and titles
	plt.xlabel('Epoch')
	plt.ylabel('Percentage')
	plt.title('Percentage of TP, TPm, FPiou over total dets of that epoch for each Epoch')
	
	#settings epoch axis to have only discrete values
	ax = plt.gca()
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))

	#saving
	plt.legend()
	plt.grid(True)
	#plt.show()
	plt.savefig(f'{base_path}.png', format='png', bbox_inches='tight')
	plt.clf()
