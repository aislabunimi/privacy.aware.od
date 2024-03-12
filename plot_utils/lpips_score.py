import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_lpips_score(lpips_score_log_path, lpips_save_name):
	epochs = []
	lpips = []
	with open(lpips_score_log_path, 'r') as file:
    		for line in file:
        		parts = line.split()
        		if len(parts) == 2:
            			epoch, lpips_score = parts
            			epochs.append(int(epoch))
            			lpips.append(float(lpips_score))

	# Creo plot
	plt.figure(figsize=(15, 10))
	plt.plot(epochs, lpips, linestyle='-', color='b', label='LPIPS score', marker='o')
	#plt.axhline(y=0.6, color='r', linestyle='-.', label='Plain Unet')
	
	# config
	ax = plt.gca()
	ax.tick_params(axis='both', which='major', labelsize=20)
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	
	plt.title('LPIPS score Over Epochs', fontsize=20)
	plt.xlabel('Epoch', fontsize=20)
	plt.ylabel('Value', fontsize=20)
	plt.legend(loc='upper right', fontsize=20)
	plt.grid(True)
	
	plt.savefig(lpips_save_name, format='png', bbox_inches='tight')
	plt.clf()
