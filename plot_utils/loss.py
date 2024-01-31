import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_model_loss(loss_log_path, loss_save_name):
	#Liste vuote
	epochs = []
	train_losses = []
	test_losses = []
	# Salvo il contenuto del file nelle liste
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
	plt.plot(epochs, train_losses, linestyle='-', color='b', label='Train Loss', marker='o')
	plt.plot(epochs, test_losses, linestyle='-', color='r', label='Validation Loss', marker='x')
	
	# config
	ax = plt.gca()
	ax.tick_params(axis='both', which='major', labelsize=14)
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	
	plt.title('Training and Validation Loss Over Epochs', fontsize=20)
	plt.xlabel('Epoch', fontsize=18)
	plt.ylabel('Value', fontsize=18)
	plt.legend(loc='upper right', fontsize=16)
	plt.grid(True)
	
	plt.savefig(loss_save_name, format='png', bbox_inches='tight')
	plt.clf()
