import pandas as pd

validation_ap = pd.read_csv('./results_csv/validation_ap.csv')

test_ap = pd.read_csv('./results_csv/testing_ap.csv')
validation_msssim = pd.read_csv('./results_csv/validation_backward.csv')
testing_msssim = pd.read_csv('./results_csv/testing_backward.csv')

#print(type(validation_ap.loc[validation_ap["Setting"] == '1pos1neg', 'AP$_{50}$'].iloc[0]))


table = ''
for n in range(0, 5):
    table += f'$n={n}$'

    ap = validation_ap
    msssim = validation_msssim
    for p in range(1, 5):
        index = str(p) + "pos" + str(n) +"neg"

        table += (f'&{round(ap.loc[ap["Setting"] == index, "AP"].iloc[0])}'
                  f'&{round(ap.loc[ap["Setting"] == index, "AP$_{50}$"].iloc[0])} '
                  f'&{round(msssim.loc[msssim["Setting"]==index, "MS-SSIM"].iloc[0]*100)}')
    ap = test_ap
    msssim = testing_msssim
    for p in range(1, 5):
        index = str(p) + "pos" + str(n) +"neg"
        table += (f'&{round(ap.loc[ap["Setting"] == index, "AP"].iloc[0])}'
                  f'&{round(ap.loc[ap["Setting"] == index, "AP$_{50}$"].iloc[0])} '
                  f'&{round(msssim.loc[msssim["Setting"]==index, "MS-SSIM"].iloc[0]*100)}')
    table += '\\\\\n'

table += ('\\hline \n \\small All prop.&\\multicolumn{12}{c||}{ $\\text{AP}=' + str(round(validation_ap[validation_ap["Setting"] == 'all_proposals'].iloc[0].iloc[1])) +
          '\\quad\\text{AP}_{50}=' + str(round(validation_ap[validation_ap["Setting"] == 'all_proposals'].iloc[0].iloc[3])) +
          ' \\quad \\text{MS}= ' + str(round(validation_ap[validation_ap["Setting"] == 'all_proposals'].iloc[0].iloc[2])) + '$}&' )
table += ('\\multicolumn{12}{c}{ $\\text{AP}=' + str(round(test_ap[test_ap["Setting"] == 'all_proposals'].iloc[0].iloc[1])) +
          '\\quad\\text{AP}_{50}=' + str(round(test_ap[test_ap["Setting"] == 'all_proposals'].iloc[0].iloc[3]))  +
          ' \\quad\\text{MS}= ' + str(round(test_ap[test_ap["Setting"] == 'all_proposals'].iloc[0].iloc[2])) + '$}\\\\\n' )

table += (' \\small TaskNet &\\multicolumn{12}{c||}{ $\\text{AP}=' + str(round(validation_ap[validation_ap["Setting"] == 'tasknet'].iloc[0].iloc[1])) +
          '\\quad\\text{AP}_{50}=' + str(round(validation_ap[validation_ap["Setting"] == 'tasknet'].iloc[0].iloc[3])) + '$}&' )
table += ('\\multicolumn{12}{c}{ $\\text{AP}=' + str(round(test_ap[test_ap["Setting"] == 'tasknet'].iloc[0].iloc[1])) +
          '\\quad\\text{AP}_{50}=' + str(round(test_ap[test_ap["Setting"] == 'tasknet'].iloc[0].iloc[3])) +'$}\\\\' )

print(table)
