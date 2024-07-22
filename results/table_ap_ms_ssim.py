import pandas as pd

validation_ap = pd.read_csv('./results_csv/validation_ap.csv')

test_ap = pd.read_csv('./results_csv/testing_ap.csv')
validation_msssim = pd.read_csv('./results_csv/validation_backward.csv')
testing_msssim = pd.read_csv('./results_csv/testing_backward.csv')

#print(type(validation_ap.loc[validation_ap["Setting"] == '1pos1neg', 'AP$_{50}$'].iloc[0]))
print('ciao')

table = ''
for n in range(0, 5):
    table += f'{n}'

    ap = validation_ap
    msssim = validation_msssim
    for p in range(1, 5):
        index = str(p) + "pos" + str(n) +"neg"
        print(index)
        table += (f'&{round(ap.loc[ap["Setting"] == index, "AP$_{50}$"].iloc[0])}('
                  f'{round(ap.loc[ap["Setting"] == index, "AP$_{75}$"].iloc[0])}) '
                  f'&{round(msssim.loc[msssim["Setting"]==index, "MS-SSIM"].iloc[0]*100)}')
    ap = test_ap
    msssim = testing_msssim
    for p in range(1, 5):
        index = str(p) + "pos" + str(n) +"neg"
        print(index)
        table += (f'&{round(ap.loc[ap["Setting"] == index, "AP$_{50}$"].iloc[0])}('
                  f'{round(ap.loc[ap["Setting"] == index, "AP$_{75}$"].iloc[0])}) '
                  f'&{round(msssim.loc[msssim["Setting"]==index, "MS-SSIM"].iloc[0]*100)}')
    table += '\\\\\n'
print(table)
