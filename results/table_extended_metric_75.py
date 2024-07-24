import pandas as pd

validation_ap = pd.read_csv('./results_csv/validation_iou75_custom_metric.csv')

test_ap = pd.read_csv('./results_csv/testing_iou75_custom_metric.csv')

table = ''
for n in range(0, 5):
    table += f'$n={n}$'

    ap = validation_ap

    for p in range(1, 5):
        index = str(p) + "pos" + str(n) +"neg"

        table += (f'&{round(ap.loc[ap["Setting"] == index, "TP_1"].iloc[0])}'
                  f'&{round(ap.loc[ap["Setting"] == index, "FPiou_1"].iloc[0])} ')
    ap = test_ap

    for p in range(1, 5):
        index = str(p) + "pos" + str(n) +"neg"

        table += (f'&{round(ap.loc[ap["Setting"] == index, "TP_1"].iloc[0])}'
                  f'&{round(ap.loc[ap["Setting"] == index, "FPiou_1"].iloc[0])}')
    table += '\\\\\n'
table += ('\\hline All prop.&\\multicolumn{8}{c||}{ $\\text{TP}=' + str(round(validation_ap[validation_ap["Setting"] == 'all_proposals'].iloc[0].iloc[1])) +
          ' \\quad \\text{BFD}= ' + str(round(validation_ap[validation_ap["Setting"] == 'all_proposals'].iloc[0].iloc[3])) + '$}&' )
table += ('\\multicolumn{8}{c}{ $\\text{TP}=' + str(round(test_ap[test_ap["Setting"] == 'all_proposals'].iloc[0].iloc[1])) +
          ' \\quad\\text{BFD}= ' + str(round(test_ap[test_ap["Setting"] == 'all_proposals'].iloc[0].iloc[3])) + '$}\\\\\n' )

table += ('TaskNet &\\multicolumn{8}{c||}{ $\\text{TP}=' + str(round(validation_ap[validation_ap["Setting"] == 'tasknet'].iloc[0].iloc[1])) +
          ' \\quad \\text{BFD}= ' + str(round(validation_ap[validation_ap["Setting"] == 'tasknet'].iloc[0].iloc[3])) + '$}&' )
table += ('\\multicolumn{8}{c}{ $\\text{TP}=' + str(round(test_ap[test_ap["Setting"] == 'tasknet'].iloc[0].iloc[1])) +
          ' \\quad\\text{BFD}= ' + str(round(test_ap[test_ap["Setting"] == 'tasknet'].iloc[0].iloc[3])) + '$}\\\\\n' )

print(table)
