import pandas as pd
import math
def scalePostures(inputDF,classnum, bodylen= 150, midpointx=200, midpointy=400):
    nExamples, nColumns = inputDF.shape
    inputDF = inputDF.astype(float)

    dataFrame = inputDF[0:0]
    # body length between neck keypoint and midhip, useful for normalisation

    for i in range(nExamples):
        row = inputDF.iloc[[i]]
        bodylengthIN = math.hypot(row.iloc[:, 2].values - row.iloc[:, 16].values,
                                   row.iloc[:, 3].values - row.iloc[:, 17].values)

        differencePCT = bodylengthIN / bodylen
        # print(differencePCT)
        #scale each point in row with the reference

        scaledRow = differencePCT * row

        #align each point
        midhipINX, midhipINY = scaledRow.iloc[:, 16].values, scaledRow.iloc[:, 17].values
        # print(midhipINX, midhipINY)
        # print('----------------------')
        #maybe change to other way around?
        x_diff = midpointx - midhipINX
        y_diff = midpointy - midhipINY
        # print(x_diff, y_diff)


        scaledRow.iloc[: ,::2] = x_diff + scaledRow.iloc[: ,::2].values
        scaledRow.iloc[: ,1::2]= y_diff + scaledRow.iloc[: ,1::2].values
        alignedRow = scaledRow
        dataFrame = pd.concat([dataFrame, alignedRow])

    dataFrame['class'] = classnum


    return dataFrame

#0,1,2
straight = pd.read_csv('c3-straight-full.csv')
back = pd.read_csv('c3-back-full.csv')
forward = pd.read_csv('c3-forward-full.csv')

straight = straight.drop(columns=['framenum'])
back = back.drop(columns=['framenum'])
forward = forward.drop(columns=['framenum'])

straight =scalePostures(straight,0)
back =scalePostures(back,1)
forward =scalePostures(forward,2)

dfmerged = pd.concat([straight, back, forward], ignore_index=True)

df = dfmerged[['neckX', 'neckY', 'RshoulderX', 'RshoulderY', 'RelbowX','RelbowY', 'RwristX', 'RwristY', 'RkneeX', 'RkneeY', 'RankleX', 'RankleY', 'class']]

df.to_csv('c3-set-full.csv', index=False)
