from utils import reconstruct_data
import pandas as pd

################ Sirve para plotaer las categorias del dataset

df = pd.read_csv('/mnt/d/home-2/Documentos/master/practica-deusto-tech/TFM/MyTabsyn/CorVAE/data/shoppers/dataset.csv')
columns = list(df.columns)

def changeN(list):
    print("[", end='', sep='')
    for i in list:
        print('"',i,'",', end='', sep='')
    print("]", end='', sep='')
    print("")

names1 = [columns[i] for i in [0,1,2,3,4,5,6,7,8,9]]
names2 = [columns[i] for i in [10,11,12,13,14,15,16]]

changeN(names1)
changeN(names2)
print(columns[17])

