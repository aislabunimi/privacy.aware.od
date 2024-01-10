import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
#Liste vuote
epochs = []
train_losses = []
test_losses = []
font = {#'family' : 'arial',
        #'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

# Salvo il contenuto del file nelle liste
with open('loss_log.txt', 'r') as file:
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
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.savefig('loss.png', format='png', bbox_inches='tight')
