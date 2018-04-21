import pandas as pd
path = 'F:/alicloth/z_rank/out/pant.csv'
path1 = 'F:/alicloth/z_rank/out1/pant.csv'
coat = pd.read_csv(path)
coat['shuxing'] = 'pant_length_labels'
coat.rename(columns={'0': 'predict'}, inplace=True)
coat['Predict'] = coat['predict']
coat.drop('predict', axis=1, inplace=True)
coat.to_csv(path1, index=None)
