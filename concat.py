import pandas as pd
import os
path = 'F:/alicloth/z_rank/out1'
pathlist = os.listdir(path)
coat = pd.read_csv(path + '/' + 'coat.csv')
collar = pd.read_csv(path + '/' + 'collar.csv')
lapel = pd.read_csv(path + '/' + 'lapel.csv')
neck = pd.read_csv(path + '/' + 'neck.csv')
neckline = pd.read_csv(path + '/' + 'neckline.csv')
pant = pd.read_csv(path + '/' + 'pant.csv')
skirt = pd.read_csv(path + '/' + 'skirt.csv')
sleeve = pd.read_csv(path + '/' + 'sleeve.csv')
#lisname = [coat, collar, lapel, neck, neckline, pant, skirt, sleeve]
lisname2 = [collar, neckline, skirt, sleeve, neck, coat, lapel, pant]
all = pd.concat(lisname2)
all.to_csv('F:/alicloth/z_rank/out1' + '/resultall1.csv', index=None)
