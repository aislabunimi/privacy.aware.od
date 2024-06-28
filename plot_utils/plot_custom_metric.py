import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import re

def plot_custom_metric(file_list, file_save_list, all_classes, five_classes):
   #list containing all files from custom metric
   for file_path, file_save in zip(file_list, file_save_list):
      pattern = r'([0-9]+\.[0-9]+)'
      matches = re.findall(pattern, file_save)
      iou_value = float(matches[0])
      score_value = float(matches[1])
      if not os.path.exists(file_path):
         print(f"The file '{file_path}' was not found. Skipping relative custom metric plotting.")
         plt.close()
         return
      with open(file_path, 'r') as json_file:
         data = json.load(json_file)
      #getting data, calculating percentage of tp, fp and fpiou over total_detections
      epochs = [entry['epoch'] for entry in data]
      if all_classes:
         n_classes = 80
         valid_labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '27', '28', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '67', '70', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87', '88', '89', '90']#need to skip background and N/A, now i have 80 classes
      elif five_classes:
         n_classes = 5
         valid_labels=['1', '2', '3', '4', '5']
      else:
         n_classes = 1
         valid_labels=['1']
      curve_dict_TP={}
      curve_dict_FP={}
      curve_dict_TPm={}
      curve_dict_FPm={}
      curve_dict_FPiou={}
      for i in valid_labels:
         #total_detections = [entry[i]['total_positives'] for entry in data]
         TP_percentage = [(entry[i]['TP'] / entry[i]['total_positives']) if entry[i]['total_detections'] != 0 else 0 for entry in data]
         FP_percentage = [(entry[i]['FP'] / entry[i]['total_positives']) if entry[i]['total_detections'] != 0 else 0 for entry in data]
         TPm_percentage = [(entry[i]['TPm'] / entry[i]['total_positives']) if entry[i]['total_detections'] != 0 else 0 for entry in data]
         FPm_percentage = [(entry[i]['FPm'] / entry[i]['total_positives']) if entry[i]['total_detections'] != 0 else 0 for entry in data]
         FPiou_percentage = [(entry[i]['FPiou'] / entry[i]['total_positives']) if entry[i]['total_detections'] != 0 else 0 for entry in data]
         curve_dict_TP[i]=TP_percentage
         curve_dict_TPm[i]=TPm_percentage
         curve_dict_FP[i]=FP_percentage
         curve_dict_FPm[i]=FPm_percentage
         curve_dict_FPiou[i]=FPiou_percentage
      #plotting now the curves
      #mean for all classes
      
      # get length list, is equal to number of epoch
      list_length = len(next(iter(curve_dict_TP.values())))
      # Initialize resulting curves
      TP_curve = [0] * list_length
      FP_curve = [0] * list_length
      TPm_curve = [0] * list_length
      FPm_curve = [0] * list_length
      FPiou_curve = [0] * list_length
      # Sum value for value for all classes
      for values in curve_dict_TP.values():
         TP_curve = [sum(x) for x in zip(TP_curve, values)]
      for values in curve_dict_FP.values():
         FP_curve = [sum(x) for x in zip(FP_curve, values)]
      for values in curve_dict_TPm.values():
         TPm_curve = [sum(x) for x in zip(TPm_curve, values)]
      for values in curve_dict_FPm.values():
         FPm_curve = [sum(x) for x in zip(FPm_curve, values)]
      for values in curve_dict_FPiou.values():
         FPiou_curve = [sum(x) for x in zip(FPiou_curve, values)]
      # compute the mean value for all classes
      TP_curve=[value / n_classes for value in TP_curve]
      FP_curve=[value / n_classes for value in FP_curve]
      TPm_curve=[value / n_classes for value in TPm_curve]
      FPm_curve=[value / n_classes for value in FPm_curve]
      FPiou_curve=[value / n_classes for value in FPiou_curve]
      
      plt.figure(figsize=(15, 10))
      plt.plot(epochs, TP_curve, label='TP', color='blue', linewidth=3.0)
      plt.plot(epochs, FP_curve, label='FP', color='red', linewidth=3.0)
      #plt.plot(epochs, TPm_curve, label='TPm', color='green', linewidth=3.0) #less important
      #plt.plot(epochs, FPm_curve, label='FPm', color='orange', linewidth=3.0)
      plt.plot(epochs, FPiou_curve, label='FPiou', color='black', linewidth=3.0)
      #label and titles
      plt.xlabel('Epoch', fontsize=30, fontweight='bold')
      plt.ylabel('Value', fontsize=30, fontweight='bold')
      plt.title(f'(IoU={iou_value}, Score={score_value}) TP, FP, TPm, FPm, FPiou Over Epochs', fontsize=29, fontweight='bold')
      #settings epoch axis to have only discrete values
      ax = plt.gca()
      ax.tick_params(axis='both', which='major', labelsize=35)
      ax.xaxis.set_major_locator(MaxNLocator(integer=True))
      #saving
      legend = plt.legend(loc='best', framealpha=0.5, fontsize=28)
      plt.grid(True)
      plt.savefig(file_save, format='png', bbox_inches='tight')
      plt.clf()
