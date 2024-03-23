import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
def read_backward_val(loss_log_path):
    epochs = []
    train_losses = []
    val_losses = []

    if not os.path.exists(loss_log_path):
        print(f"The file '{loss_log_path}' was not found. Skipping MS-SSIM score plotting.")
        return

    with open(loss_log_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 3:
            	epoch, train_loss, val_loss = parts
            	epochs.append(int(epoch))
            	train_losses.append(float(train_loss))
            	val_losses.append(float(val_loss))

    return epochs, val_losses

def plot_normalized_curves(file_paths, labels, save_name):
    all_data = []
    min_val = float(10000)
    max_val = float(-1)
    for file_path in file_paths:
        data = read_backward_val(file_path)
        if data:
            all_data.append(data)
            min_val = min(min_val, min(data[1]))
            max_val = max(max_val, max(data[1]))

    if not all_data:
        print("No data available for plotting.")
        return

    plt.figure(figsize=(15, 10))
    for data, label in zip(all_data, labels):
        epochs, val_loss = data
        normalized_val_loss = [(val - min_val) / (max_val - min_val) for val in val_loss]
        plt.plot(epochs, normalized_val_loss, linestyle='-', linewidth=3.0, label=label)

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=35)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title('Normalized Val Loss Over Epochs', fontsize=29, fontweight='bold')
    plt.xlabel('Epoch', fontsize=30, fontweight='bold')
    plt.ylabel('Normalized Value', fontsize=30, fontweight='bold')
    plt.legend(loc='best', framealpha=0.5, fontsize=28)
    plt.grid(True)

    plt.savefig(save_name, format='png', bbox_inches='tight')
    plt.clf()

def plot_curves(file_paths, labels, save_name):
    all_data = []
    for file_path in file_paths:
        data = read_backward_val(file_path)
        if data:
            all_data.append(data)

    if not all_data:
        print("No data available for plotting.")
        return

    plt.figure(figsize=(15, 10))
    for data, label in zip(all_data, labels):
        epochs, val_loss = data
        plt.plot(epochs, val_loss, linestyle='-', linewidth=3.0, label=label)

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=35)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title('Val Loss Over Epochs', fontsize=29, fontweight='bold')
    plt.xlabel('Epoch', fontsize=30, fontweight='bold')
    plt.ylabel('Normalized Value', fontsize=30, fontweight='bold')
    plt.legend(loc='best', framealpha=0.5, fontsize=28)
    plt.grid(True)

    plt.savefig(save_name, format='png', bbox_inches='tight')
    plt.clf()

file_paths = ["file1.txt", "file2.txt", "file3.txt"] 
labels = ["Curve 1", "Curve 2", "Curve 3"]
save_name = "backward_val_loss_comparison_normalized.png"
plot_normalized_curves(file_paths, labels, save_name)
save_name = "backward_val_loss_comparison.png"
plot_curves(file_paths, labels, save_name)
