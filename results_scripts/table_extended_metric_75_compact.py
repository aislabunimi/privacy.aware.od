import pandas as pd

validation_data = pd.read_csv('results_csv/validation_iou75_custom_metric.csv')

test_data = pd.read_csv('results_csv/testing_iou75_custom_metric.csv')

#print(type(validation_ap.loc[validation_ap["Setting"] == '1pos1neg', 'AP$_{50}$'].iloc[0]))


table = ''
for n in range(0, 5):
    table += f'$n={n}$'


    for data in [validation_data, test_data]:
        for metric in ["TP_1", "FPiou_1"]:
            for p in range(1, 5):
                index = str(p) + "pos" + str(n) +"neg"
                table += (f'&{round(data.loc[data["Setting"] == index, metric].iloc[0])}')
    table += '\\\\\n'

table += ('\\hline  All&\\multicolumn{4}{c|}{' + str(round(validation_data[validation_data["Setting"] == 'all_proposals'].iloc[0].iloc[1])) +'}'
          ' &\\multicolumn{4}{c||}{' + str(round(validation_data[validation_data["Setting"] == 'all_proposals'].iloc[0].iloc[3])) + '}&' )
table += ('\\multicolumn{4}{c|}{' + str(round(test_data[test_data["Setting"] == 'all_proposals'].iloc[0].iloc[1])) +'}'
          ' &\\multicolumn{4}{c}{ ' + str(round(test_data[test_data["Setting"] == 'all_proposals'].iloc[0].iloc[3]))+'}\\\\\n')
table += ('TN&\\multicolumn{4}{c|}{' + str(round(validation_data[validation_data["Setting"] == 'tasknet'].iloc[0].iloc[1])) +'}'
        ' &\\multicolumn{4}{c||}{' + str(round(validation_data[validation_data["Setting"] == 'tasknet'].iloc[0].iloc[3])) + '}&' )
table += ('\\multicolumn{4}{c|}{' + str(round(test_data[test_data["Setting"] == 'tasknet'].iloc[0].iloc[1])) +'}'
' &\\multicolumn{4}{c}{ ' + str(round(test_data[test_data["Setting"] == 'tasknet'].iloc[0].iloc[3]))+'}\\\\\n')



print(table)
