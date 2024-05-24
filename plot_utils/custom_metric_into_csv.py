import json
import csv
import os

# compute percentages of index
def compute_percentages(entry):
    total_positives = entry.get("total_positives", 0)
    if total_positives == 0:
        return 0, 0, 0

    TP = (entry.get("TP", 0) / total_positives)*100
    TPm = (entry.get("TPm", 0) / total_positives)*100
    FPiou = (entry.get("FPiou", 0) / total_positives)*100
    
    return TP, TPm, FPiou

input_file_names = ['0.5_MAE'] #, '0.7_MAE', '0.8_MAE', '0.9_MAE']
#input_file_names = ['all_proposals', '1pos0neg', '1pos1neg', '1pos2neg', '1pos3neg', '1pos4neg', '2pos0neg', '2pos1neg', '2pos2neg', '2pos3neg', '2pos4neg', '3pos0neg', '3pos1neg', '3pos2neg', '3pos3neg', '3pos4neg', '4pos0neg', '4pos1neg', '4pos2neg', '4pos3neg', '4pos4neg']
input_file_paths = [] #then create paths
for name in input_file_names:
   path = f'/home/math0012/Tesi_magistrale/PAPER_Results/{name}/forward/results_fw/iou0.5_score0.75.json'
   input_file_paths.append(path)
epoch = 50
output_file_path = 'validation_custom_metrics.csv'
#keys_to_extract = ["1", "2", "3", "4", "5"] #for multiclasses
keys_to_extract = ["1"]

csv_data = [["file_name"]]
# Add column names for TP, TPm, and FPiou based on keys_to_extract
for key in keys_to_extract:
    csv_data[0].append(f"TP_{key}")
    csv_data[0].append(f"TPm_{key}")
    csv_data[0].append(f"FPiou_{key}")
    
#csv_data = [["file_name", "TP", "TPm", "FPiou"]]

# for every file
for input_file_path, name in zip(input_file_paths, input_file_names):
    with open(input_file_path, 'r') as f:
        data = json.load(f)
    TP_dict = {}
    TPm_dict = {}
    FPiou_dict = {}
    for key in keys_to_extract:
       # extract right epoch
       for d in data:
          if d.get("epoch") == epoch:
             entry = d.get(key, {})
             break
       
       TP, TPm, FPiou = compute_percentages(entry)
       TP = f"{TP:.2f}"
       TPm = f"{TPm:.2f}"
       FPiou = f"{FPiou:.2f}"
       TP_dict[key] = TP
       TPm_dict[key] = TPm
       FPiou_dict[key] = FPiou

    row = [name]
    for key in keys_to_extract:
        row.extend([TP_dict.get(key, ""), TPm_dict.get(key, ""), FPiou_dict.get(key, "")])
    csv_data.append(row)

with open(output_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(csv_data)

# folder names for each experiments
input_file_names = ['tasknet_1norm_myresize', 'all_proposals', '1pos0neg', '1pos1neg', '1pos2neg', '1pos3neg', '1pos4neg', '2pos0neg', '2pos1neg', '2pos2neg', '2pos3neg', '2pos4neg', '3pos0neg', '3pos1neg', '3pos2neg', '3pos3neg', '3pos4neg', '4pos0neg', '4pos1neg', '4pos2neg', '4pos3neg', '4pos4neg']
input_file_paths = [] #for creating paths
for name in input_file_names:
   path = f'/home/math0012/Tesi_magistrale/PAPER_Results/{name}/test_results/iou0.5_score0.75.json'
   input_file_paths.append(path)
epoch = 0
output_file_path = 'testing_custom_metric.csv'

csv_data = [["file_name"]]
# Add column names for TP, TPm, and FPiou based on keys_to_extract
for key in keys_to_extract:
    csv_data[0].append(f"TP_{key}")
    csv_data[0].append(f"TPm_{key}")
    csv_data[0].append(f"FPiou_{key}")

# for every file
for input_file_path, name in zip(input_file_paths, input_file_names):
    with open(input_file_path, 'r') as f:
        data = json.load(f)
    TP_dict = {}
    TPm_dict = {}
    FPiou_dict = {}
    for key in keys_to_extract:
       # extract right epoch
       for d in data:
          if d.get("epoch") == epoch:
             entry = d.get(key, {})
             break
       
       TP, TPm, FPiou = compute_percentages(entry)
       TP = f"{TP:.2f}"
       TPm = f"{TPm:.2f}"
       FPiou = f"{FPiou:.2f}"
       TP_dict[key] = TP
       TPm_dict[key] = TPm
       FPiou_dict[key] = FPiou

    row = [name]
    for key in keys_to_extract:
        row.extend([TP_dict.get(key, ""), TPm_dict.get(key, ""), FPiou_dict.get(key, "")])
    csv_data.append(row)

with open(output_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(csv_data)

print('done')

