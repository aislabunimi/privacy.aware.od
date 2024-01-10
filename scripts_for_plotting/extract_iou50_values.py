def extract_values(lines):
    epoch = int(lines[0].split()[-1])
    ap_line = [line for line in lines if 'Average Precision' in line and 'IoU=0.50      | area=   all | maxDets=100' in line]
    ar_line = [line for line in lines if 'Average Recall' in line and 'IoU=0.50:0.95 | area=   all | maxDets=100' in line]
    
    ap_value = float(ap_line[0].split('=')[-1].strip())
    ar_value = float(ar_line[0].split('=')[-1].strip())
    
    return epoch, ap_value, ar_value


def process_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    results = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("AP for Epoch"):
            epoch, ap_value, ar_value = extract_values(lines[i:i+14])  # 14 linee per ogni epoca
            results.append((epoch, ap_value, ar_value))
            i += 14  # salto le linee fino alla prossima epoca
        else:
            i += 1

    with open(output_file, 'w') as f:
        for result in results:
            f.write(f"{result[0]} {result[1]} {result[2]}\n")


if __name__ == "__main__":
    input_file = "ap_log.txt"
    output_file = "ap_iou50.txt"

    process_file(input_file, output_file)

