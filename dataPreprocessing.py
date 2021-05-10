import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import csv

#synthesized data
dfBAD = pd.read_csv('badOutputs/CSV/c1bad(5000).csv')
dfGOOD = pd.read_csv('goodOutputs/CSV/c1good(5000).csv')
#real - test data
dftestgood= pd.read_csv('goodOutputs/CSV/ytc1good.csv')
dftestbad= pd.read_csv('badOutputs/CSV/ytc1bad.csv')

#add class column 1= incorrect, 0 = correct form
df1 = pd.DataFrame({"class": [1,1,1,1,1]}) #incorrect form
df0 = pd.DataFrame({"class": [0,0,0,0,0]}) #correct form
dftestgood = dftestgood.drop(columns=['framenum'])
dftestbad = dftestbad.drop(columns=['framenum'])
dftestgood = dftestgood.join(df1)
dftestbad = dftestbad.join(df0)


#merge good and bad for train/test
dftrainmerged = pd.concat([dfBAD, dfGOOD], ignore_index=True)
dftestmerged = pd.concat([dftestgood, dftestbad], ignore_index=True)



## Preproccess

#Get NaN count per part
#print(df.isna().sum())

#Remove NaN columns and useless columns
#DF column format = 'neckX', 'neckY', 'RshoulderX', 'RshoulderY', 'RelbowX','RelbowY', 'RwristX', 'RwristY', 'RkneeX', 'RkneeY', 'RankleX', 'RankleY', 'class'
traindf = dftrainmerged[['neckX', 'neckY', 'RshoulderX', 'RshoulderY', 'RelbowX','RelbowY', 'RwristX', 'RwristY', 'RkneeX', 'RkneeY', 'RankleX', 'RankleY', 'class']]
testdf = dftestmerged[['neckX', 'neckY', 'RshoulderX', 'RshoulderY', 'RelbowX','RelbowY', 'RwristX', 'RwristY', 'RkneeX', 'RkneeY', 'RankleX', 'RankleY', 'class']]

#save training and test df's to csv files
traindf.to_csv('trainsetNN(10000).csv', index=False)
#testdf.to_csv('testsetNN.csv', index=False)



