import pandas as pd

validation_ap = pd.read_csv('./results_csv/validation_iou75_custom_metric.csv')

test_ap = pd.read_csv('./results_csv/testing_iou75_custom_metric.csv')

#print(type(validation_ap.loc[validation_ap["Setting"] == '1pos1neg', 'AP$_{50}$'].iloc[0]))
print('ciao')

table = ''
for n in range(0, 5):
    table += f'{n}'

    ap = validation_ap

    for p in range(1, 5):
        index = str(p) + "pos" + str(n) +"neg"
        print(index)
        table += (f'&{round(ap.loc[ap["Setting"] == index, "TP_1"].iloc[0])}'
                  f'&{round(ap.loc[ap["Setting"] == index, "FPiou_1"].iloc[0])} ')
    ap = test_ap

    for p in range(1, 5):
        index = str(p) + "pos" + str(n) +"neg"
        print(index)
        table += (f'&{round(ap.loc[ap["Setting"] == index, "TP_1"].iloc[0])}'
                  f'&{round(ap.loc[ap["Setting"] == index, "FPiou_1"].iloc[0])}')
    table += '\\\\\n'
print(table)
