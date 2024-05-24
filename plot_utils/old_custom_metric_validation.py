import json
import csv
import os

# percentages of indexes
def compute_percentages(entry):
    total_positives = entry.get("total_positives", 0)
    if total_positives == 0:
        return 0, 0, 0

    TP = (entry.get("TP", 0) / total_positives)*100
    TPm = (entry.get("TPm", 0) / total_positives)*100
    FPiou = (entry.get("FPiou", 0) / total_positives)*100
    
    return TP, TPm, FPiou

# input file folder
input_file_names = ['all_proposals', '1pos0neg', '1pos1neg', '1pos2neg', '1pos3neg', '1pos4neg', '2pos0neg', '2pos1neg', '2pos2neg', '2pos3neg', '2pos4neg', '3pos0neg', '3pos1neg', '3pos2neg', '3pos3neg', '3pos4neg', '4pos0neg', '4pos1neg', '4pos2neg', '4pos3neg', '4pos4neg']
input_file_paths = [] #then create paths
for name in input_file_names:
   path = f'/home/math0012/Tesi_magistrale/PAPER_Results/{name}/forward/results_fw/iou0.5_score0.75.json'
   input_file_paths.append(path)
epoch = 50
output_file_path = 'old_validation_custom_metrics.csv'

csv_data = [["file_name", "TP", "TPm", "FPiou"]]

#tasknet, exception case
tasknet_path='/home/math0012/Tesi_magistrale/PAPER_Results/tasknet_1norm_myresize/AP_indoor_results_A100/iou0.5_score0.75.json'
with open(tasknet_path, 'r') as f:
   data = json.load(f)
entry = next((d for d in data if d.get("epoch") == 1), None) #epoch=1
TP, TPm, FPiou = compute_percentages(entry)
TP = f"{TP:.2f}"
TPm = f"{TPm:.2f}"
FPiou = f"{FPiou:.2f}"
csv_data.append(['tasknet', TP, TPm, FPiou])

# for every file
for input_file_path, name in zip(input_file_paths, input_file_names):
    with open(input_file_path, 'r') as f:
        data = json.load(f)
    
    # extract right epoch
    entry = next((d for d in data if d.get("epoch") == epoch), None)

    TP, TPm, FPiou = compute_percentages(entry)
    TP = f"{TP:.2f}"
    TPm = f"{TPm:.2f}"
    FPiou = f"{FPiou:.2f}"

    csv_data.append([name, TP, TPm, FPiou])



with open(output_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(csv_data)

print("done")

