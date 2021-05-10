import pandas as pd
import math
import numpy as np

#good posture csv
DFGOOD = pd.read_csv('goodOutputs/CSV/ytc1good.csv')
DFGOOD = DFGOOD.replace(0, np.nan)
DFGOOD= DFGOOD.drop(columns=['framenum'])
DFGOOD = DFGOOD.drop(DFGOOD.columns[2::3], axis=1)

#input posture csv
DFBAD = pd.read_csv('badOutputs/CSV/ytc1bad.csv')
DFBAD = DFBAD.replace(0, np.nan)
DFBAD= DFBAD.drop(columns=['framenum'])
DFBAD = DFBAD.drop(DFBAD.columns[2::3], axis=1)



#scales posture to a specified bodylen, and midpoint (x,y)
#df inputs should be processed before use(remove unnecessary columns etc)
def scalePostures(inputDF,classnum, bodylen= 150, midpointx=200, midpointy=400):
    nExamples, nColumns = inputDF.shape
    inputDF = inputDF.astype(float)

    dataFrame = inputDF[0:0]
    # body length between neck keypoint and midhip, useful for normalisation

    for i in range(nExamples):
        row = inputDF.iloc[[i]]
        bodylengthIN = math.hypot(row.iloc[:, 2].values - row.iloc[:, 16].values,
                                   row.iloc[:, 3].values - row.iloc[:, 17].values)
        print(bodylengthIN)
        differencePCT = bodylengthIN / bodylen
        # print(differencePCT)
        #scale each point in row with the reference

        scaledRow = differencePCT * row

        #align each point
        midhipINX, midhipINY = scaledRow.iloc[:, 16].values, scaledRow.iloc[:, 17].values
        # print(midhipINX, midhipINY)
        # print('----------------------')

        x_diff = midpointx - midhipINX
        y_diff = midpointy - midhipINY
        # print(x_diff, y_diff)


        scaledRow.iloc[: ,::2] = x_diff + scaledRow.iloc[: ,::2].values
        scaledRow.iloc[: ,1::2]= y_diff + scaledRow.iloc[: ,1::2].values
        alignedRow = scaledRow
        dataFrame = pd.concat([dataFrame, alignedRow])

        dataFrame['class'] =  classnum


    return dataFrame


#function to preprocess and mix both good/bad scaled csv's
def merge(gdf,bdf):
    merged= pd.concat([gdf, bdf], ignore_index=True)
    df = merged[['neckX', 'neckY', 'RshoulderX', 'RshoulderY', 'RelbowX','RelbowY', 'RwristX', 'RwristY', 'RkneeX', 'RkneeY', 'RankleX', 'RankleY', 'class']]
    return df

good = scalePostures(DFGOOD,0)
bad = scalePostures(DFBAD, 1)


df = merge(good,bad)
df.to_csv('ytc1SCALED.csv', index=False)
