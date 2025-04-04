import pandas as pd

validation_ap = pd.read_csv('results_csv_5classes/validation_ap.csv')

test_ap = pd.read_csv('results_csv_5classes/testing_ap.csv')
validation_msssim = pd.read_csv('results_csv_5classes/validation_backward.csv')
testing_msssim = pd.read_csv('results_csv_5classes/testing_backward.csv')

validation_extended_50 = pd.read_csv('results_csv_5classes/validation_iou50_custom_metric.csv')
validation_extended_75 = pd.read_csv('results_csv_5classes/validation_iou75_custom_metric.csv')

testing_extended_50 = pd.read_csv('results_csv_5classes/testing_iou50_custom_metric.csv')
testing_extended_75 = pd.read_csv('results_csv_5classes/testing_iou75_custom_metric.csv')

#print(type(validation_ap.loc[validation_ap["Setting"] == '1pos1neg', 'AP$_{50}$'].iloc[0]))


table = ''
indexes = [i for i in range(2, 4)] + ['allprop', 'tasknet']
for n in indexes:



    ap = validation_ap
    msssim = validation_msssim
    extended_50 = validation_extended_50
    extended_75 = validation_extended_75

    if isinstance(n, int):
        index = str(n) + "pos" + str(n) +"neg"
        table += f'\\small ${n, n}$'
    else:
        index = n
        table += f'\\small All' if index == 'allprop' else '\\small TN'
    print(index)
    table += (f'&{round(ap.loc[ap["Setting"] == index, "AP"].iloc[0])}&'
                  f'{round(ap.loc[ap["Setting"] == index, "AP$_{50}$"].iloc[0])} '
              f'&{round(extended_75.loc[extended_75["Setting"] == index, ["TP_1", "TP_2", "TP_3", "TP_4"]].iloc[0].mean())}'
              f'&{round(extended_75.loc[extended_75["Setting"] == index, ["FPiou_1", "FPiou_2", "FPiou_3", "FPiou_4"]].iloc[0].mean())}')
    if index != 'tasknet':
        table += f'&{round(msssim.loc[msssim["Setting"]==index, "MS-SSIM"].iloc[0]*100)}'
    else:
        table += f'&--'
    ap = test_ap
    msssim = testing_msssim
    extended_50= testing_extended_50
    extended_75 = testing_extended_75
    table += (f'&{round(ap.loc[ap["Setting"] == index, "AP"].iloc[0])}'
              f'&{round(ap.loc[ap["Setting"] == index, "AP$_{50}$"].iloc[0])} '
              f'&{round(extended_75.loc[extended_75["Setting"] == index, ["TP_1", "TP_2", "TP_3", "TP_4"]].iloc[0].mean())}'
              f'&{round(extended_75.loc[extended_75["Setting"] == index, ["FPiou_1", "FPiou_2", "FPiou_3", "FPiou_4"]].iloc[0].mean())}')
    if index != 'tasknet':
        table += f'&{round(msssim.loc[msssim["Setting"]==index, "MS-SSIM"].iloc[0]*100)}'
    else:
        table += f'&--'
    if n == 3:
        table += '\\\\\\hline\n'
    else:
        table += '\\\\\n'

print(table)
