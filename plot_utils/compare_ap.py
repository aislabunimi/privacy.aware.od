import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
def plot_compare_between_two_ap(ap_log_path, ap_log_to_compare_path, ap_model_name, ap_to_compare_model_name, plotted_comparison_save_path, ap_plot_title):
   epochs_model = []
   ap_model = []
   recall_model = []
   epochs_model_compare = []
   ap_model_compare = []
   recall_model_compare = []
   if not os.path.exists(ap_log_path):
      print(f"The file '{ap_log_path}' was not found. Skipping compare AP and AR plotting.")
      plt.close()
      return
   with open(ap_log_path, 'r') as file:
      for line in file:
         parts = line.split()
         if len(parts) == 3:
            epoch, ap_v, recall_v = parts
            epochs_model.append(int(epoch))
            ap_model.append(float(ap_v))
            recall_model.append(float(recall_v))
   if not os.path.exists(ap_log_to_compare_path):
      print(f"The file '{ap_log_to_compare_path}' was not found. Skipping compare AP and AR plotting.")
      plt.close()
      return
   with open(ap_log_to_compare_path, 'r') as file:
      for line in file:
         parts = line.split()
         if len(parts) == 3:
            epoch, ap_v, recall_v = parts
            epochs_model_compare.append(int(epoch))
            ap_model_compare.append(float(ap_v))
            recall_model_compare.append(float(recall_v))

   plt.figure(figsize=(15, 10))
   plt.plot(epochs_model, ap_model, linestyle='-', color='b', label=f'AP {ap_model_name}', linewidth=3.0)
   plt.plot(epochs_model, recall_model, linestyle='-', color='r', label=f'Recall {ap_model_name}', linewidth=3.0)
   plt.plot(epochs_model_compare, ap_model_compare, linestyle='--', color='b', label=f'AP {ap_to_compare_model_name}', linewidth=3.0)
   plt.plot(epochs_model_compare, recall_model_compare, linestyle='--', color='r', label=f'Recall {ap_to_compare_model_name}', linewidth=3.0)
   plt.title(ap_plot_title, fontsize=30, fontweight='bold')
   
   ax = plt.gca()
   ax.tick_params(axis='both', which='major', labelsize=35)
   ax.xaxis.set_major_locator(MaxNLocator(integer=True))
   
   plt.xlabel('Epoch', fontsize=30, fontweight='bold')
   plt.ylabel('Value', fontsize=30, fontweight='bold')
   plt.legend(loc='best', framealpha=0.5, fontsize=28)
   plt.grid(True)
   
   plt.savefig(plotted_comparison_save_path, format='png', bbox_inches='tight')
   plt.clf()
