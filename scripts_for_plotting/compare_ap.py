import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
epochs_model = []
ap_model = []
recall_model = []
epochs_model_compare = []
ap_model_compare = []
recall_model_compare = []
font = {#'family' : 'arial',
        #'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

#Config
ap_log = 'ap_large.txt'
ap_log_name="Entropy SGD"
ap_log_compare = 'ap_sgd_large.txt'
ap_log_compare_name="Plain SGD"
#ap_log_compare = 'ap_large_sgd_65k.txt'

with open(ap_log, 'r') as file:
    for line in file:
        parts = line.split()
        if len(parts) == 3:
            epoch, ap_v, recall_v = parts
            epochs_model.append(int(epoch))
            ap_model.append(float(ap_v))
            recall_model.append(float(recall_v))

with open(ap_log_compare, 'r') as file:
    for line in file:
        parts = line.split()
        if len(parts) == 3:
            epoch, ap_v, recall_v = parts
            epochs_model_compare.append(int(epoch))
            ap_model_compare.append(float(ap_v))
            recall_model_compare.append(float(recall_v))

plt.figure(figsize=(15, 10))
plt.plot(epochs_model, ap_model, linestyle='-', color='b', label=f'AP {ap_log_name}', marker='o')
plt.plot(epochs_model, recall_model, linestyle='-', color='r', label=f'Recall {ap_log_name}', marker='o')

plt.plot(epochs_model_compare, ap_model_compare, linestyle='--', color='b', label=f'AP {ap_log_compare_name}', marker='x')
plt.plot(epochs_model_compare, recall_model_compare, linestyle='--', color='r', label=f'Recall {ap_log_compare_name}', marker='x')

#Plotto i limiti massimi raggiungibili dalla tasknet
#Area ALL
#plt.axhline(y=0.429, color='b', linestyle='-.', label='Tasknet AP')
#plt.axhline(y=0.547, color='r', linestyle='-.', label='Tasknet Recall')
#plt.title('(IoU=0.50:0.95, area=all, maxDets=100) AP and Recall Over Epochs')
#Area Large
plt.axhline(y=0.844, color='b', linestyle='-.', label='Tasknet AP')
plt.axhline(y=0.867, color='r', linestyle='-.', label='Tasknet Recall')
plt.title('(IoU=0.50:0.95, area=large, maxDets=100) AP and Recall Over Epochs')

ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.savefig('compare_ap.png', format='png', bbox_inches='tight')
