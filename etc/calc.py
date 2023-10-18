import pandas as pd

target_file = './permute_Cross_c10_b1'

df = pd.read_csv(target_file+'.csv')

df.T.describe().to_csv(target_file+'_describe.csv')