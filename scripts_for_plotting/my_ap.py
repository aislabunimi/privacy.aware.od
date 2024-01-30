import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

epochs = []
ap = []
recall = []
font = {#'family' : 'arial',
        #'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

# Dati dal file
with open('my_ap_iou50.txt', 'r') as file:
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

#Plotto i limiti massimi raggiungibili dalla tasknet
#Area ALL
#plt.axhline(y=0.429, color='b', linestyle='-.', label='Tasknet AP')
#plt.axhline(y=0.547, color='r', linestyle='-.', label='Tasknet Recall')
#plt.title('(IoU=0.50:0.95, area=all, maxDets=100) AP and Recall Over Epochs')
#Area Large
plt.axhline(y=1, color='b', linestyle='-.', label='Unet AP')
plt.axhline(y=0.585, color='r', linestyle='-.', label='Unet Recall')
plt.title('(IoU=0.50, area=all, maxDets=100) AP and Recall Over Epochs')

ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend(loc='upper left') #la metto in alto a sx
plt.grid(True)

plt.savefig('my_ap.png', format='png', bbox_inches='tight')

