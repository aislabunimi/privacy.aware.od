import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
def plot_model_loss(loss_log_path, loss_save_name):
	#Liste vuote
	epochs = []
	train_losses = []
	test_losses = []
	# Salvo il contenuto del file nelle liste
	if not os.path.exists(loss_log_path):
           print(f"The file '{loss_log_path}' was not found. Skipping loss plotting.")
           plt.close()
           return
	with open(loss_log_path, 'r') as file:
    		for line in file:
        		parts = line.split()
        		if len(parts) == 3:
            			epoch, train_loss, test_loss = parts
            			epochs.append(int(epoch))
            			train_losses.append(float(train_loss))
            			test_losses.append(float(test_loss))

	# Creo plot
	plt.figure(figsize=(15, 10))
	plt.plot(epochs, train_losses, linestyle='-', color='b', label='Train Loss', linewidth=3.0)
	plt.plot(epochs, test_losses, linestyle='-', color='r', label='Val Loss', linewidth=3.0)
	
	# config
	ax = plt.gca()
	ax.tick_params(axis='both', which='major', labelsize=35)
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	
	plt.title('Training and Validation Loss Over Epochs', fontsize=30, fontweight='bold')
	plt.xlabel('Epoch', fontsize=30, fontweight='bold')
	plt.ylabel('Value', fontsize=30, fontweight='bold')
	plt.legend(loc='best', framealpha=0.5, fontsize=28)
	plt.grid(True)
	
	plt.savefig(loss_save_name, format='png', bbox_inches='tight')
	plt.clf()
