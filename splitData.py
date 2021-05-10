import numpy as np
import pandas as pd


#split training and test from dataset 0.7
df = pd.read_csv('c1-set-full.csv')
df = df.sample(frac=1).reset_index(drop=True)
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.8
train = df[msk]
test = df[~msk]
train = train.drop(columns=['split'])
test = test.drop(columns=['split'])
train.to_csv('train-c7.csv', index=False)
test.to_csv('test-c7.csv', index=False)