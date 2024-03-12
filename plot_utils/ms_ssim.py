import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_ms_ssim_score(ms_ssim_score_log_path, ms_ssim_save_name):
	#Liste vuote
	epochs = []
	ms_ssim = []
	# Salvo il contenuto del file nelle liste
	with open(ms_ssim_score_log_path, 'r') as file:
    		for line in file:
        		parts = line.split()
        		if len(parts) == 2:
            			epoch, ms_score = parts
            			epochs.append(int(epoch))
            			ms_ssim.append(float(ms_score))

	# Creo plot
	plt.figure(figsize=(15, 10))
	plt.plot(epochs, ms_ssim, linestyle='-', color='b', label='MS_SSIM score', marker='o')
	#plt.axhline(y=0.6, color='r', linestyle='-.', label='Plain Unet')
	
	# config
	ax = plt.gca()
	ax.tick_params(axis='both', which='major', labelsize=20)
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	
	plt.title('MS_SSIM score Over Epochs', fontsize=20)
	plt.xlabel('Epoch', fontsize=20)
	plt.ylabel('Value', fontsize=20)
	plt.legend(loc='upper right', fontsize=20)
	plt.grid(True)
	
	plt.savefig(ms_ssim_save_name, format='png', bbox_inches='tight')
	plt.clf()
