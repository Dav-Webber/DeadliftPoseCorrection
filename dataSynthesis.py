import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import random

#for each checkpoint
#get set of posture keypoints (good and bad separate)
#align and scale with one posture (testing input will need to be aligned to the same posture used here)
#get the mean for each joint using all the samples for the same joint
#get std of mean for each joint
#create function that generates random numbers in range of mean +- std of mean



##CSV HEADER (class = 0 for correct and 1 for incorrect)
csvHeader =     ['noseX','noseY','neckX','neckY',
                    'RshoulderX','RshoulderY', 'RelbowX','RelbowY',
                    'RwristX','RwristY', 'LshoulderX','LshoulderY',
                    'LelbowX','LelbowY', 'LwristX','LwristY',
                    'midhipX','midhipY','RhipX','RhipY',
                    'RkneeX','RkneeY','RankleX','RankleY',
                    'LhipX','LhipY','LkneeX','LkneeY',
                    'LankleX','LankleY','ReyeX','ReyeY',
                    'LeyeX','LeyeY','RearX','RearY',
                    'LearX','LearY','LbigtoeX','LbigtoeY',
                    'LsmalltoeX','LsmalltoeY','LheelX','LheelY',
                    'RbigtoeX','RbigtoeY','RsmalltoeX','RsmalltoeY',
                    'RheelX','RheelY','class']



DFREAL = pd.read_csv('badOutputs/CSV/c1bad(2.0).csv')
DFREAL = DFREAL.replace(0, np.nan)
DFREAL= DFREAL.drop(columns=['framenum'])
DFREAL = DFREAL.drop(DFREAL.columns[2::3], axis=1)


#scale and align all postures with one posture
#df inputs should be processed before use(remove unnecessary columns etc)
#scales posture to another postures location
def scalePostures2posture(inputDF, posture2scale2):
    nExamples, nColumns = inputDF.shape
    inputDF = inputDF.astype(float)
    posture2scale2 = posture2scale2.astype(float)
    dataFrame = inputDF[0:0]
    # body length between neck keypoint and midhip, useful for normalisation
    bodylengthREF = math.hypot(posture2scale2.iloc[:, 2].values - posture2scale2.iloc[:, 16].values, posture2scale2.iloc[:, 3].values - posture2scale2.iloc[:, 17].values)
    midhipREFX, midhipREFY = posture2scale2.iloc[:, 16].values,posture2scale2.iloc[:, 17].values
    for i in range(nExamples):
        row = inputDF.iloc[[i]]
        bodylengthIN = math.hypot(row.iloc[:, 2].values - row.iloc[:, 16].values,
                                   row.iloc[:, 3].values - row.iloc[:, 17].values)
        differencePCT = bodylengthIN / bodylengthREF

        #scale each point in row with the reference

        scaledRow = differencePCT * row

        #align each point
        midhipINX, midhipINY = scaledRow.iloc[:, 16].values, scaledRow.iloc[:, 17].values

        #maybe change to other way around?
        x_diff = midhipREFX - midhipINX
        y_diff = midhipREFY - midhipINY

        scaledRow.iloc[: ,::2] = x_diff + scaledRow.iloc[: ,::2].values
        scaledRow.iloc[: ,1::2]= y_diff + scaledRow.iloc[: ,1::2].values
        alignedRow = scaledRow
        dataFrame = pd.concat([dataFrame, alignedRow])
    return dataFrame

#scales posture to a specified bodylen, and midpoint (x,y)
#df inputs should be processed before use(remove unnecessary columns etc)
def scalePostures2(inputDF, bodylen= 100, midpointx=50, midpointy=400):
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
    return dataFrame

scaledDF = scalePostures2(DFREAL)
# print(scaledDF)
#array of lists = inputData, so 0th element of all lists = nose  (all data should be scaled and alligned before input here)
#num2gen = to specify how many rows of data to generate
#binary = 0 if correct and 1 if incorrect, form
def augmentData(inputDF, num2gen, checkpointNum, binary):
    #the number of different postures in input and columns
    nExamples, nColumns = inputDF.shape
    meanDF = inputDF.mean(axis=0)
    stdDF = inputDF.std(axis=0)
    lowR = meanDF - stdDF
    highR = meanDF + stdDF
    # print("LOW ranges: ")
    # print(lowR)
    # print("High ranges: ")
    # print(highR)
    print('Checkpoint '+str(checkpointNum))
    for i in range(num2gen):
        gData = random.uniform(lowR,highR)
        #1 = incorrect form, 0 = correct form
        classNum = pd.Series([binary])
        gData = gData.append(classNum)
        #prints row to csv file
        wr.writerow(gData)
        print(i)

with open("badOutputs/CSV/c1bad(5000).csv", "a") as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(csvHeader)
    #synthesize 1000 rows of data for c1
    augmentData(scaledDF, 5000, 1, 1)


