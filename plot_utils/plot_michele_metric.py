import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
def plot_michele_metric(file_list, file_save_list):
	#list containing all files from michele metric
	for file_path, file_save in zip(file_list, file_save_list):
		# first i remove the json extension
		base_path, _ = os.path.splitext(file_path)
		with open(file_path, 'r') as json_file:
    			data = json.load(json_file)
		#getting data, calculating percentage of tp, fp and fpiou over total_detections
		epochs = [entry['epoch'] for entry in data]
		total_detections = [entry['total_detections'] for entry in data]
		TP_percentage = [(entry['TP'] / entry['total_detections']) if entry['total_detections'] != 0 else 0 for entry in data]
		#FP_percentage = [(entry['FP'] / entry['total_detections']) for entry in data]
		#gli FP non ha senso printarli, perché le pred della faster sono sempre con label 1 di persona, non potrà mai fare una pred con label di background!
		TPm_percentage = [(entry['TPm'] / entry['total_detections']) if entry['total_detections'] != 0 else 0 for entry in data]
		FPiou_percentage = [(entry['FPiou'] / entry['total_detections']) if entry['total_detections'] != 0 else 0 for entry in data]
	
		#plotting now the curves
		plt.figure(figsize=(15, 10))
		plt.plot(epochs, TP_percentage, label='TP Percentage', color='blue')
		#plt.plot(epochs, FP_percentage, label='FP Percentage', color='red')
		plt.plot(epochs, TPm_percentage, label='TPm Percentage', color='green')
		plt.plot(epochs, FPiou_percentage, label='FPiou Percentage', color='red')
	
		#label and titles
		plt.xlabel('Epoch', fontsize=18)
		plt.ylabel('Percentage', fontsize=18)
		plt.title('Percentage of TP, TPm, FPiou over total dets of that epoch for each Epoch', fontsize=20)
	
		#settings epoch axis to have only discrete values
		ax = plt.gca()
		ax.tick_params(axis='both', which='major', labelsize=14)
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))

		#saving
		plt.legend(loc='upper right', fontsize=16)
		plt.grid(True)
		plt.savefig(file_save, format='png', bbox_inches='tight')
		plt.clf()
