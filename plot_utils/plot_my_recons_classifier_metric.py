import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
def plot_my_recons_classifier_metric(file_path, file_save):
   with open(file_path, 'r') as json_file:
      data = json.load(json_file)
   #getting data, calculating percentage of tp, fp and fpiou over total_detections
   epochs = [entry['epoch'] for entry in data]
   total_detections = [entry['total'] for entry in data]
   zero_percentage = [(entry['0'] / entry['total']) if entry['total'] != 0 else 0 for entry in data]
   one_percentage = [(entry['1'] / entry['total']) if entry['total'] != 0 else 0 for entry in data]
   two_percentage = [(entry['2'] / entry['total']) if entry['total'] != 0 else 0 for entry in data]
   three_percentage = [(entry['3'] / entry['total']) if entry['total'] != 0 else 0 for entry in data]
   """ Cosa rappresenta ciascun valore:
   0 rappresentano img con solo rumore
   1 img con contorni persone, non identificabili
   2 img con presenza persone riconoscibile facilmente ma difficilmente identificabili
   3 img con persone riconoscibili e identificabili    
   """
   
   plt.figure(figsize=(15, 10))
   plt.plot(epochs, zero_percentage, label='Noise', color='black')
   plt.plot(epochs, one_percentage, label='Person Silhoutte', color='blue')
   plt.plot(epochs, two_percentage, label='Hardly Identifiable person', color='green')
   plt.plot(epochs, three_percentage, label='Identifiable person', color='red')
   
   #label and titles
   plt.xlabel('Epoch', fontsize=18)
   plt.ylabel('Percentage', fontsize=18)
   plt.title('Percentage of each class occurence over total dataset each Epoch', fontsize=20)
   
   #settings epoch axis to have only discrete values
   ax = plt.gca()
   ax.tick_params(axis='both', which='major', labelsize=14)
   ax.xaxis.set_major_locator(MaxNLocator(integer=True))
   
   #saving
   plt.legend(loc='lower left', fontsize=16)
   plt.grid(True)
   plt.savefig(file_save, format='png', bbox_inches='tight')
   plt.clf()
