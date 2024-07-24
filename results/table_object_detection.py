import pandas as pd

validation_ap = pd.read_csv('./results_csv_5classes/validation_ap.csv')

test_ap = pd.read_csv('./results_csv_5classes/testing_ap.csv')
validation_msssim = pd.read_csv('./results_csv/validation_backward.csv')
testing_msssim = pd.read_csv('./results_csv/testing_backward.csv')

validation_extended_50 = pd.read_csv('./results_csv_5classes/validation_iou50_custom_metric.csv')
validation_extended_75 = pd.read_csv('./results_csv_5classes/validation_iou75_custom_metric.csv')

testing_extended_50 = pd.read_csv('./results_csv_5classes/testing_iou50_custom_metric.csv')
testing_extended_75 = pd.read_csv('./results_csv_5classes/testing_iou75_custom_metric.csv')

#print(type(validation_ap.loc[validation_ap["Setting"] == '1pos1neg', 'AP$_{50}$'].iloc[0]))


table = ''
for n in range(2, 4):
    table += f'{n}'

    ap = validation_ap
    msssim = validation_msssim
    extended_50 = validation_extended_50
    extended_75 = validation_extended_50

    index = str(n) + "pos" + str(n) +"neg"

    table += (f'&{round(ap.loc[ap["Setting"] == index, "AP$_{50}$"].iloc[0])}('
                  f'{round(ap.loc[ap["Setting"] == index, "AP$_{75}$"].iloc[0])}) '
              f'{round(ap.loc[extended_50["Setting"] == index])}'
                  f'&{round(msssim.loc[msssim["Setting"]==index, "MS-SSIM"].iloc[0]*100)}')
    ap = test_ap
    msssim = testing_msssim
    for p in range(1, 5):
        index = str(p) + "pos" + str(n) +"neg"
        table += (f'&{round(ap.loc[ap["Setting"] == index, "AP$_{50}$"].iloc[0])}('
                  f'{round(ap.loc[ap["Setting"] == index, "AP$_{75}$"].iloc[0])}) '
                  f'&{round(msssim.loc[msssim["Setting"]==index, "MS-SSIM"].iloc[0]*100)}')
    table += '\\\\\n'

table += ('\\hline All prop.&\\multicolumn{8}{c||}{ $\\text{TP}=' + str(round(validation_ap[validation_ap["Setting"] == 'all_proposals'].iloc[0].iloc[1])) +
          '(' + str(round(validation_ap[validation_ap["Setting"] == 'all_proposals'].iloc[0].iloc[3])) + ')' +
          ' \\quad \\text{MS}= ' + str(round(validation_ap[validation_ap["Setting"] == 'all_proposals'].iloc[0].iloc[2])) + '$}&' )
table += ('\\multicolumn{8}{c}{ $\\text{TP}=' + str(round(test_ap[test_ap["Setting"] == 'all_proposals'].iloc[0].iloc[1])) +
          '(' + str(round(test_ap[test_ap["Setting"] == 'all_proposals'].iloc[0].iloc[3])) + ')' +
          ' \\quad\\text{MS}= ' + str(round(test_ap[test_ap["Setting"] == 'all_proposals'].iloc[0].iloc[2])) + '$}\\\\' )

print(table)
