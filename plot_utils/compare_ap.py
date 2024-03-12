import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_compare_between_two_ap(ap_log_path, ap_log_to_compare_path, ap_model_name, ap_to_compare_model_name, plotted_comparison_save_path, ap_plot_title):
	epochs_model = []
	ap_model = []
	recall_model = []
	epochs_model_compare = []
	ap_model_compare = []
	recall_model_compare = []

	with open(ap_log_path, 'r') as file:
    		for line in file:
        		parts = line.split()
        		if len(parts) == 3:
            			epoch, ap_v, recall_v = parts
           	 		epochs_model.append(int(epoch))
            			ap_model.append(float(ap_v))
            			recall_model.append(float(recall_v))

	with open(ap_log_to_compare_path, 'r') as file:
    		for line in file:
       		 	parts = line.split()
       		 	if len(parts) == 3:
            			epoch, ap_v, recall_v = parts
           		 	epochs_model_compare.append(int(epoch))
            			ap_model_compare.append(float(ap_v))
            			recall_model_compare.append(float(recall_v))

	plt.figure(figsize=(15, 10))
	plt.plot(epochs_model, ap_model, linestyle='-', color='b', label=f'AP {ap_model_name}', marker='o')
	plt.plot(epochs_model, recall_model, linestyle='-', color='r', label=f'Recall {ap_model_name}', marker='o')

	plt.plot(epochs_model_compare, ap_model_compare, linestyle='--', color='b', label=f'AP {ap_to_compare_model_name}', marker='x')
	plt.plot(epochs_model_compare, recall_model_compare, linestyle='--', color='r', label=f'Recall {ap_to_compare_model_name}', marker='x')

	plt.title(ap_plot_title, fontsize=20)
	ax = plt.gca()
	ax.tick_params(axis='both', which='major', labelsize=20)
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))

	plt.xlabel('Epoch', fontsize=20)
	plt.ylabel('Value', fontsize=20)
	plt.legend(loc='upper left', fontsize=20)
	plt.grid(True)

	plt.savefig(plotted_comparison_save_path, format='png', bbox_inches='tight')
	plt.clf()
