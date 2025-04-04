import pandas as pd

validation_ap = pd.read_csv('results_csv/validation_ap.csv')

test_ap = pd.read_csv('results_csv/testing_ap.csv')
validation_msssim = pd.read_csv('results_csv/validation_backward.csv')
testing_msssim = pd.read_csv('results_csv/testing_backward.csv')

#print(type(validation_ap.loc[validation_ap["Setting"] == '1pos1neg', 'AP$_{50}$'].iloc[0]))


table = ''
for n in range(0, 5):
    table += f'\\small $n={n}$'

    ap = validation_ap
    msssim = validation_msssim
    for metric, dataframe in zip(['AP', "AP$_{50}$","MS-SSIM"], [ap, ap, msssim]):

        for p in range(1, 5):
            index = str(p) + "pos" + str(n) +"neg"

            table += (f'&{round(dataframe.loc[dataframe["Setting"] == index, metric].iloc[0] if metric != "MS-SSIM" else dataframe.loc[dataframe["Setting"] == index, metric].iloc[0]*100)}')
    ap = test_ap
    msssim = testing_msssim
    for metric, dataframe in zip(['AP', "AP$_{50}$","MS-SSIM"], [ap, ap, msssim]):
        for p in range(1, 5):
            index = str(p) + "pos" + str(n) +"neg"
            table += (f'&{round(dataframe.loc[dataframe["Setting"] == index, metric].iloc[0] if metric != "MS-SSIM" else dataframe.loc[dataframe["Setting"] == index, metric].iloc[0] *100)}')
    table += '\\\\\n'

table += ('\\hline \n \\small All prop.&\\multicolumn{4}{c|}{ ' + str(round(validation_ap[validation_ap["Setting"] == 'all_proposals'].iloc[0].iloc[1])) + '}' +
          '&\\multicolumn{4}{c|}{ ' + str(round(validation_ap[validation_ap["Setting"] == 'all_proposals'].iloc[0].iloc[3])) + '}' +
          ' &\\multicolumn{4}{c||}{ ' + str(round(validation_msssim[validation_msssim["Setting"] == 'all_proposals'].iloc[0].iloc[2]*100)) + '}&' )
table += ('\\multicolumn{4}{c|}{ ' + str(round(test_ap[test_ap["Setting"] == 'all_proposals'].iloc[0].iloc[1])) + '}' +
          '&\\multicolumn{4}{c|}{ ' + str(round(test_ap[test_ap["Setting"] == 'all_proposals'].iloc[0].iloc[3]))  +'}' +
          ' &\\multicolumn{4}{c}{  ' + str(round(testing_msssim[testing_msssim["Setting"] == 'all_proposals'].iloc[0].iloc[2]*100)) + '}\\\\\n' )

table += (' \\small TaskNet &\\multicolumn{4}{c|}{ ' + str(round(validation_ap[validation_ap["Setting"] == 'tasknet'].iloc[0].iloc[1])) +'}' +
          '&\\multicolumn{4}{c|}{' + str(round(validation_ap[validation_ap["Setting"] == 'tasknet'].iloc[0].iloc[3])) + '}' +
          '&\\multicolumn{4}{c||}{100}&')
table += ('\\multicolumn{4}{c|}{'  + str(round(test_ap[test_ap["Setting"] == 'tasknet'].iloc[0].iloc[1])) +'}&' +
          '\\multicolumn{4}{c|}{ ' + str(round(test_ap[test_ap["Setting"] == 'tasknet'].iloc[0].iloc[3])) +'}&' +
          '\\multicolumn{4}{c}{ 100}\\\\')

print(table)
