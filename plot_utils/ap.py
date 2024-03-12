import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_ap(ap_log_path, ap_save_name, best_ap_value_for_comparison=None, best_recall_value_for_comparison=None, model_name='model', ap_plot_title='AP and recall'):
	epochs = []
	ap = []
	recall = []
	# Dati dal file
	with open(ap_log_path, 'r') as file:
    		for line in file:
        		parts = line.split()
        		if len(parts) == 3:
            			epoch, ap_v, recall_v = parts
            			epochs.append(int(epoch))
            			ap.append(float(ap_v))
            			recall.append(float(recall_v))

	plt.figure(figsize=(15, 10))
	plt.plot(epochs, ap, linestyle='-', color='b', label='AP', marker='o')
	plt.plot(epochs, recall, linestyle='-', color='r', label='Recall', marker='x')

	if best_ap_value_for_comparison is not None:
		plt.axhline(y=best_ap_value_for_comparison, color='b', linestyle='-.', label=f'{model_name} AP')
	if best_recall_value_for_comparison is not None:
		plt.axhline(y=best_recall_value_for_comparison, color='r', linestyle='-.', label=f'{model_name} Recall')
	plt.title(ap_plot_title, fontsize=20)
	
	ax = plt.gca()
	ax.tick_params(axis='both', which='major', labelsize=20)
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	
	plt.xlabel('Epoch', fontsize=20)
	plt.ylabel('Value', fontsize=20)
	plt.legend(loc='upper left', fontsize=20) #la metto in alto a sx
	plt.grid(True)
	
	plt.savefig(ap_save_name, format='png', bbox_inches='tight')
	plt.clf()

