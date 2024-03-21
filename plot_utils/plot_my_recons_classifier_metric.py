import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
def plot_my_recons_classifier_metric(file_path, file_save):
   if not os.path.exists(file_path):
      print(f"The file '{file_path}' was not found. Skipping my classifier metric plotting.")
      plt.close()
      return
   with open(file_path, 'r') as json_file:
      data = json.load(json_file)
   #getting data, calculating percentage of tp, fp and fpiou over total_detections
   epochs = [entry['epoch'] for entry in data]
   total_detections = [entry['total'] for entry in data]
   zero_percentage = [(entry['0'] / entry['total']) if entry['total'] != 0 else 0 for entry in data]
   one_percentage = [(entry['1'] / entry['total']) if entry['total'] != 0 else 0 for entry in data]
   two_percentage = [(entry['2'] / entry['total']) if entry['total'] != 0 else 0 for entry in data]
   three_percentage = [(entry['3'] / entry['total']) if entry['total'] != 0 else 0 for entry in data]
   """ What every value represents:
   0 images with only noiserappresentano img con solo rumore
   1 images with people silhouette, not identifiable
   2 images not identifiable but it's easy to see the presence of a person
   3 images with person identifiable
   """  
   plt.figure(figsize=(15, 10))
   plt.plot(epochs, zero_percentage, label='Noise', color='black', linewidth=3.0)
   plt.plot(epochs, one_percentage, label='Person Silhoutte', color='blue', linewidth=3.0)
   plt.plot(epochs, two_percentage, label='Hardly Identifiable person', color='green', linewidth=3.0)
   plt.plot(epochs, three_percentage, label='Identifiable person', color='red',linewidth=3.0)
   
   #label and titles
   plt.xlabel('Epoch', fontsize=30, fontweight='bold')
   plt.ylabel('Probability', fontsize=30, fontweight='bold')
   plt.title('Probability for each class over dataset each Epoch', fontsize=30, fontweight='bold')
   
   #settings epoch axis to have only discrete values
   ax = plt.gca()
   ax.tick_params(axis='both', which='major', labelsize=35)
   ax.xaxis.set_major_locator(MaxNLocator(integer=True))
   
   #saving
   plt.legend(loc='best', framealpha=0.5, fontsize=28)
   plt.grid(True)
   plt.savefig(file_save, format='png', bbox_inches='tight')
   plt.clf()
   
def plot_my_recons_classifier_metric_probs(file_path, file_save):
   if not os.path.exists(file_path):
      print(f"The file '{file_path}' was not found. Skipping my classifier probs metric plotting.")
      plt.close()
      return
   with open(file_path, 'r') as json_file:
      data = json.load(json_file)
   #getting data, calculating percentage of tp, fp and fpiou over total_detections
   epochs = [entry['epoch'] for entry in data]
   total_detections = [entry['total'] for entry in data]
   zero_percentage = [(entry['prob0tot'] / entry['total']) if entry['total'] != 0 else 0 for entry in data]
   one_percentage = [(entry['prob1tot'] / entry['total']) if entry['total'] != 0 else 0 for entry in data]
   two_percentage = [(entry['prob2tot'] / entry['total']) if entry['total'] != 0 else 0 for entry in data]
   three_percentage = [(entry['prob3tot'] / entry['total']) if entry['total'] != 0 else 0 for entry in data]
 
   plt.figure(figsize=(15, 10))
   plt.plot(epochs, zero_percentage, label='Noise', color='black', linewidth=3.0)
   plt.plot(epochs, one_percentage, label='Person Silhoutte', color='blue', linewidth=3.0)
   plt.plot(epochs, two_percentage, label='Hardly Identifiable person', color='green', linewidth=3.0)
   plt.plot(epochs, three_percentage, label='Identifiable person', color='red',linewidth=3.0)
   
   #label and titles
   plt.xlabel('Epoch', fontsize=30, fontweight='bold')
   plt.ylabel('Probability', fontsize=30, fontweight='bold')
   plt.title('Probability for each class over dataset each Epoch', fontsize=30, fontweight='bold')
   
   #settings epoch axis to have only discrete values
   ax = plt.gca()
   ax.tick_params(axis='both', which='major', labelsize=35)
   ax.xaxis.set_major_locator(MaxNLocator(integer=True))
   
   #saving
   plt.legend(loc='best', framealpha=0.5, fontsize=28)
   plt.grid(True)
   plt.savefig(file_save, format='png', bbox_inches='tight')
   plt.clf()
