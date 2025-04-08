from matplotlib import pyplot as plt

files = ['all_proposals.txt', '4pos4neg.txt']

base_path = 'mia_results/'
for file_name in files:
    with open(base_path + file_name) as file:
        old_attacker = [float(line.rstrip().split(' ')[1]) for line in file]

    with open(base_path + 'mia_' + file_name) as file:
        mia_attacker = [float(line.rstrip().split(' ')[1]) for line in file]
    print(old_attacker, mia_attacker)
    plt.plot([i for i in range(len(old_attacker))], old_attacker, label='(Old) MAE attacker')
    plt.plot([i for i in range(len(mia_attacker))], mia_attacker,label='(New) MIA attacker')
    plt.legend()
    plt.title(file_name)
    plt.show()
