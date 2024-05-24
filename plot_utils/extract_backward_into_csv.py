import csv

def extract_val_loss(file_path, target_epoch):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            epoch, value = line.strip().split()
            if int(epoch) == target_epoch:
                return float(value)
    return None

def extract_sim_metrics(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        MS_SSIM_value = None
        LPIPS_value = None
        for line in lines:
            if line.startswith("MS_SSIM"):
                MS_SSIM_value = float(line.split(":")[-1].strip())
            elif line.startswith("LPIPS"):
                LPIPS_value = float(line.split(":")[-1].strip())
        return MS_SSIM_value, LPIPS_value

input_file_names = ['all_proposals', '1pos0neg', '1pos1neg', '1pos2neg', '1pos3neg', '1pos4neg', '2pos0neg', '2pos1neg', '2pos2neg', '2pos3neg', '2pos4neg', '3pos0neg', '3pos1neg', '3pos2neg', '3pos3neg', '3pos4neg', '4pos0neg', '4pos1neg', '4pos2neg', '4pos3neg', '4pos4neg']
input_path_val = [] #then create paths
input_path_sim = []
for name in input_file_names:
   path_val = f'/home/math0012/Tesi_magistrale/PAPER_Results/{name}/backward/results_bw/only_val_loss_log_batch1.txt'
   path_sim = f'/home/math0012/Tesi_magistrale/PAPER_Results/{name}/backward/results_bw/similarity_log.txt'
   input_path_val.append(path_val)
   input_path_sim.append(path_sim)
output_file_path = 'validation_backward.csv'
epoch = 80

csv_data = [["file_name", "val_loss", "MS_SSIM", "LPIPS"]]

# each file
for name, path_val, path_sim in zip(input_file_names, input_path_val, input_path_sim):
    val = extract_val_loss(path_val, epoch)
    MS_SSIM, LPIPS = extract_sim_metrics(path_sim)
    val = f"{val:.4f}"
    MS_SSIM = f"{MS_SSIM:.4f}"
    LPIPS = f"{LPIPS:.4f}"
    csv_data.append([name, val, MS_SSIM, LPIPS])

with open(output_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(csv_data)

input_path_val = [] #then create paths
input_path_sim = []
for name in input_file_names:
   path_val = f'/home/math0012/Tesi_magistrale/PAPER_Results/{name}/test_results/only_test_loss_bw.txt'
   path_sim = f'/home/math0012/Tesi_magistrale/PAPER_Results/{name}/test_results/similarity_log.txt'
   input_path_val.append(path_val)
   input_path_sim.append(path_sim)
output_file_path = 'testing_backward.csv'
epoch = 0

csv_data = [["file_name", "val_loss", "MS_SSIM", "LPIPS"]]

# each  file
for name, path_val, path_sim in zip(input_file_names, input_path_val, input_path_sim):
    val = extract_val_loss(path_val, epoch)
    MS_SSIM, LPIPS = extract_sim_metrics(path_sim)
    val = f"{val:.4f}"
    MS_SSIM = f"{MS_SSIM:.4f}"
    LPIPS = f"{LPIPS:.4f}"
    csv_data.append([name, val, MS_SSIM, LPIPS])

with open(output_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(csv_data)

print('done')

