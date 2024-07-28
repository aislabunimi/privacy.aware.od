import csv

def extract_AP_50(file_path, target_epoch):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if line.startswith("AP for Epoch") and int(line.split()[-1]) == target_epoch:
                # Iterate through the lines until we find the desired line
                while i < len(lines):
                    line = lines[i]
                    if line.startswith(" Average Precision  (AP) @[ IoU=0.50:0.50 | area=   all | maxDets=100 ]"):
                        return float(line.split("=")[-1].strip())*100
                    i += 1
                break
    return None

def extract_AR_50(file_path, target_epoch):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if line.startswith("AP for Epoch") and int(line.split()[-1]) == target_epoch:
                # Iterate through the lines until we find the desired line
                while i < len(lines):
                    line = lines[i]
                    if line.startswith(" Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=100 ]"):
                        return float(line.split("=")[-1].strip())*100
                    i += 1
                break
    return None
    
def extract_AP_75(file_path, target_epoch):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if line.startswith("AP for Epoch") and int(line.split()[-1]) == target_epoch:
                # Iterate through the lines until we find the desired line
                while i < len(lines):
                    line = lines[i]
                    if line.startswith(" Average Precision  (AP) @[ IoU=0.75:0.75 | area=   all | maxDets=100 ]"):
                        return float(line.split("=")[-1].strip())*100
                    i += 1
                break
    return None

def extract_AP(file_path, target_epoch):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if line.startswith("AP for Epoch") and int(line.split()[-1]) == target_epoch:
                # Iterate through the lines until we find the desired line
                while i < len(lines):
                    line = lines[i]
                    if line.startswith(" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]"):
                        return float(line.split("=")[-1].strip())*100
                    i += 1
                break
    return None

def extract_AR_75(file_path, target_epoch):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if line.startswith("AP for Epoch") and int(line.split()[-1]) == target_epoch:
                # Iterate through the lines until we find the desired line
                while i < len(lines):
                    line = lines[i]
                    if line.startswith(" Average Recall     (AR) @[ IoU=0.75:0.75 | area=   all | maxDets=100 ]"):
                        return float(line.split("=")[-1].strip())*100
                    i += 1
                break
    return None
def extract_AR(file_path, target_epoch):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if line.startswith("AP for Epoch") and int(line.split()[-1]) == target_epoch:
                # Iterate through the lines until we find the desired line
                while i < len(lines):
                    line = lines[i]
                    if line.startswith(" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]"):
                        return float(line.split("=")[-1].strip())*100
                    i += 1
                break
    return None
input_file_names = ['all_proposals', '1pos0neg', '1pos1neg', '1pos2neg', '1pos3neg', '1pos4neg', '2pos0neg', '2pos1neg', '2pos2neg', '2pos3neg', '2pos4neg', '3pos0neg', '3pos1neg', '3pos2neg', '3pos3neg', '3pos4neg', '4pos0neg', '4pos1neg', '4pos2neg', '4pos3neg', '4pos4neg']
input_path_ap_50 = [] #then create paths
input_path_ap_75 = []
input_path_complete=[]
for name in input_file_names:
   path_50 = f'/home/alberti/model_training/main/experiments/val_batch_1/{name}/val_results_batch1/standard_ap_iou50.txt'
   path_75 = f'/home/alberti/model_training/main/experiments/val_batch_1/{name}/val_results_batch1/standard_ap_iou75.txt'
   path_complete = f'/home/alberti/model_training/main/experiments/val_batch_1/{name}/val_results_batch1/standard_ap.txt'
   input_path_ap_50.append(path_50)
   input_path_ap_75.append(path_75)
   input_path_complete.append(path_complete)
output_file_path = 'validation_ap.csv'
epoch = 0

csv_data = [["file_name",'AP', 'AR', "AP_50", "AR_50", "AP_75", "AR_75"]]

#tasknet, exception case
AP_50 = extract_AP_50('/home/alberti/model_training/main/experiments/val_batch_1/tasknet/val_results_batch1/standard_ap_iou50.txt', 0)
AR_50 = extract_AR_50('/home/alberti/model_training/main/experiments/val_batch_1/tasknet/val_results_batch1/standard_ap_iou50.txt', 0)
AP_75 = extract_AP_75('/home/alberti/model_training/main/experiments/val_batch_1/tasknet/val_results_batch1/standard_ap_iou75.txt', 0)
AR_75 = extract_AR_75('/home/alberti/model_training/main/experiments/val_batch_1/tasknet/val_results_batch1/standard_ap_iou75.txt', 0)
AP = extract_AP('/home/alberti/model_training/main/experiments/val_batch_1/tasknet/val_results_batch1/standard_ap.txt', 0)
AR = extract_AR('/home/alberti/model_training/main/experiments/val_batch_1/tasknet/val_results_batch1/standard_ap.txt', 0)
AP_50 = f"{AP_50:.1f}"
AR_50 = f"{AR_50:.1f}"
AP_75 = f"{AP_75:.1f}"
AR_75 = f"{AR_75:.1f}"
AP = f"{AP:.1f}"
AR = f"{AR:.1f}"
csv_data.append(['tasknet',AP, AR, AP_50, AR_50, AP_75, AR_75])

#input files
for name, path_50, path_75, path_complete in zip(input_file_names, input_path_ap_50, input_path_ap_75, input_path_complete):
    # extract aps
    AP_50 = extract_AP_50(path_50, epoch)
    AR_50 = extract_AR_50(path_50, epoch)
    AP_75 = extract_AP_75(path_75, epoch)
    AR_75 = extract_AR_75(path_75, epoch)
    AP = extract_AP(path_complete, epoch)
    AR = extract_AR(path_complete, epoch)
    AP_50 = f"{AP_50:.1f}"
    AR_50 = f"{AR_50:.1f}"
    AP_75 = f"{AP_75:.1f}"
    AR_75 = f"{AR_75:.1f}"
    AP = f"{AP:.1f}"
    AR = f"{AR:.1f}"
    
    csv_data.append([name, AP, AR, AP_50, AR_50, AP_75, AR_75])

with open(output_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(csv_data)


input_path_ap_50 = [] #create paths for testing
input_path_ap_75 = []
input_path_complete = []
input_file_names.append('tasknet')
for name in input_file_names:
   path_50 = f'/home/alberti/model_training/main/experiments/{name}/test_results/standard_ap_iou50.txt'
   path_75 = f'/home/alberti/model_training/main/experiments/{name}/test_results/standard_ap_iou75.txt'
   path_complete=f'/home/alberti/model_training/main/experiments/{name}/test_results/standard_ap.txt'
   input_path_ap_50.append(path_50)
   input_path_ap_75.append(path_75)
   input_path_complete.append(path_complete)
output_file_path = 'testing_ap.csv'
epoch = 0

csv_data = [["file_name", 'AP', 'AR', "AP_50", "AR_50", "AP_75", "AR_75"]]


#input files
for name, path_50, path_75, path_complete in zip(input_file_names, input_path_ap_50, input_path_ap_75, input_path_complete):
    # extract aps
    AP_50 = extract_AP_50(path_50, epoch)
    AR_50 = extract_AR_50(path_50, epoch)
    AP_75 = extract_AP_75(path_75, epoch)
    AR_75 = extract_AR_75(path_75, epoch)
    AP =extract_AP(path_complete, epoch)
    AR= extract_AR(path_complete, epoch)
    AP_50 = f"{AP_50:.1f}"
    AR_50 = f"{AR_50:.1f}"
    AP_75 = f"{AP_75:.1f}"
    AR_75 = f"{AR_75:.1f}"
    AP = f"{AP:.1f}"
    AR= f"{AR:.1f}"
    
    csv_data.append([name, AP, AR, AP_50, AR_50, AP_75, AR_75])

with open(output_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(csv_data)


print('done')

