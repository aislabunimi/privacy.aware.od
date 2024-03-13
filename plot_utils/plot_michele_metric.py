import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import re
def plot_michele_metric(file_list, file_save_list):
	#list containing all files from michele metric
	for file_path, file_save in zip(file_list, file_save_list):
		pattern = r'([0-9]+\.[0-9]+)'
		matches = re.findall(pattern, file_path)
		iou_value = float(matches[0])
		score_value = float(matches[1])
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
		plt.plot(epochs, TP_percentage, label='TP', color='blue', linewidth=3.0)
		#plt.plot(epochs, FP_percentage, label='FP Percentage', color='red')
		plt.plot(epochs, TPm_percentage, label='TPm', color='green', linewidth=3.0)
		plt.plot(epochs, FPiou_percentage, label='FPiou', color='red', linewidth=3.0)
	
		#label and titles
		plt.xlabel('Epoch', fontsize=30, fontweight='bold')
		plt.ylabel('Value', fontsize=30, fontweight='bold')
		plt.title(f'(IoU={iou_value}, Score={score_value}) TP, TPm, FPiou Over Epochs', fontsize=29, fontweight='bold')
	
		#settings epoch axis to have only discrete values
		ax = plt.gca()
		ax.tick_params(axis='both', which='major', labelsize=35)
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))

		#saving
		legend = plt.legend(loc='best', framealpha=0.5, fontsize=28)
		plt.grid(True)
		plt.savefig(file_save, format='png', bbox_inches='tight')
		plt.clf()
