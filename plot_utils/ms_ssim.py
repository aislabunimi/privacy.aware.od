import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
def plot_ms_ssim_score(ms_ssim_score_log_path, ms_ssim_save_name):
   epochs = []
   ms_ssim = []
   if not os.path.exists(ms_ssim_score_log_path):
      print(f"The file '{ms_ssim_score_log_path}' was not found. Skipping MS-SSIM score plotting.")
      plt.close()
      return
   with open(ms_ssim_score_log_path, 'r') as file:
      for line in file:
         parts = line.split()
         if len(parts) == 2:
            epoch, ms_score = parts
            epochs.append(int(epoch))
            ms_ssim.append(float(ms_score))
   
   plt.figure(figsize=(15, 10))
   plt.plot(epochs, ms_ssim, linestyle='-', color='b', label='MS_SSIM score', linewidth=3.0)
   
   ax = plt.gca()
   ax.tick_params(axis='both', which='major', labelsize=35)
   ax.xaxis.set_major_locator(MaxNLocator(integer=True))
   
   plt.title('MS_SSIM score Over Epochs', fontsize=29, fontweight='bold')
   plt.xlabel('Epoch', fontsize=30, fontweight='bold')
   plt.ylabel('Value', fontsize=30, fontweight='bold')
   plt.legend(loc='best', framealpha=0.5, fontsize=28)
   plt.grid(True)
   
   plt.savefig(ms_ssim_save_name, format='png', bbox_inches='tight')
   plt.clf()
