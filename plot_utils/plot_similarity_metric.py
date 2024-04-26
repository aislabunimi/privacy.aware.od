import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
def plot_sim_metric(score_log_path, save_name, lb, title):
   epochs = []
   score = []
   if not os.path.exists(score_log_path):
      print(f"The file '{score_log_path}' was not found. Skipping plotting.")
      plt.close()
      return
   with open(score_log_path, 'r') as file:
      for line in file:
         parts = line.split()
         if len(parts) == 2:
            epoch, ms_score = parts
            epochs.append(int(epoch))
            score.append(float(ms_score))
   
   plt.figure(figsize=(15, 10))
   plt.plot(epochs, score, linestyle='-', color='b', label=lb, linewidth=3.0)
   
   ax = plt.gca()
   ax.tick_params(axis='both', which='major', labelsize=35)
   ax.xaxis.set_major_locator(MaxNLocator(integer=True))
   
   plt.title(title, fontsize=29, fontweight='bold')
   plt.xlabel('Epoch', fontsize=30, fontweight='bold')
   plt.ylabel('Value', fontsize=30, fontweight='bold')
   plt.legend(loc='best', framealpha=0.5, fontsize=28)
   plt.grid(True)
   
   plt.savefig(save_name, format='png', bbox_inches='tight')
   plt.clf()
