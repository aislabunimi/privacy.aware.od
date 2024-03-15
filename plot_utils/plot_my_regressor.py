import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
def plot_reconrate(reconrate_log_path, reconrate_save_name):
	#Liste vuote
	epochs = []
	values = []
	# Salvo il contenuto del file nelle liste
	if not os.path.exists(reconrate_log_path):
           print(f"The file '{reconrate_log_path}' was not found. Skipping Regressor Recon Rate plotting.")
           plt.close()
           return
	with open(reconrate_log_path, 'r') as file:
    		for line in file:
        		parts = line.split()
        		if len(parts) == 2:
            			epoch, value = parts
            			epochs.append(int(epoch))
            			values.append(float(value))

	# Creo plot
	plt.figure(figsize=(15, 10))
	plt.plot(epochs, values, linestyle='-', color='b', label='Train Loss', linewidth=3.0)
	
	# config
	ax = plt.gca()
	ax.tick_params(axis='both', which='major', labelsize=35)
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	
	plt.title('Regression Reconstruction Rate Over Epochs', fontsize=30, fontweight='bold')
	plt.xlabel('Epoch', fontsize=30, fontweight='bold')
	plt.ylabel('Value', fontsize=30, fontweight='bold')
	plt.legend(loc='best', framealpha=0.5, fontsize=28)
	plt.grid(True)
	
	plt.savefig(reconrate_save_name, format='png', bbox_inches='tight')
	plt.clf()
