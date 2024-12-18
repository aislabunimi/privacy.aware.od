import os
def extract_values(lines, split_line_value, coco_iou_modified):
    epoch = int(lines[0].split()[split_line_value])
    if coco_iou_modified==50: #if AP is computed on iou 0.50:0.50
    	ap_line = [line for line in lines if 'Average Precision' in line and 'IoU=0.50:0.50 | area=   all | maxDets=100' in line]
    	ar_line = [line for line in lines if 'Average Recall' in line and 'IoU=0.50:0.50 | area=   all | maxDets=100' in line]
    elif coco_iou_modified==75: #if AP is computed on iou 0.75:0.75
    	ap_line = [line for line in lines if 'Average Precision' in line and 'IoU=0.75:0.75 | area=   all | maxDets=100' in line]
    	ar_line = [line for line in lines if 'Average Recall' in line and 'IoU=0.75:0.75 | area=   all | maxDets=100' in line]
    else:
    	ap_line = [line for line in lines if 'Average Precision' in line and 'IoU=0.50      | area=   all | maxDets=100' in line]
    	ar_line = [line for line in lines if 'Average Recall' in line and 'IoU=0.50:0.95 | area=   all | maxDets=100' in line]
    
    ap_value = float(ap_line[0].split('=')[-1].strip())
    ar_value = float(ar_line[0].split('=')[-1].strip())
    
    return epoch, ap_value, ar_value


def extract_ap(input_file, output_file, standard_ap=True, coco_iou_modified=None):
    if not os.path.exists(input_file):
       print(f"The file '{input_file}' was not found. Skipping AP and AR extraction.")
       return
    with open(input_file, 'r') as f:
        lines = f.readlines()

    if standard_ap:
    	split_line_value=-1 #needed only to pick the epoch value from txt
    else:
    	split_line_value=3
    results = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("AP for Epoch"): #14 lines for every epoch
            epoch, ap_value, ar_value = extract_values(lines[i:i+14], split_line_value, coco_iou_modified) 
            results.append((epoch, ap_value, ar_value))
            i += 14  #skip to next epoch
        else:
            i += 1

    with open(output_file, 'w') as f:
        for result in results:
            f.write(f"{result[0]} {result[1]} {result[2]}\n")
