import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# using frame data from openpose body 25
df = pd.read_csv('badOutputs/CSV/outputBAD2.csv')
df = df.replace(0, np.nan)

totalFrames = df['framenum'].max()
wristsMax = np.sum(df[['LwristY', 'RwristY']].max()) / 2
shouldersMax = np.sum(df[['LshoulderY', 'RshoulderY']].max()) / 2
elbowsMax = np.sum(df[['LelbowY', 'RelbowY']].max()) / 2
hipMax = df[['midhipY']].max()


# calculates and returns which side rY or lY has the highest probability for a known part
def largest(part, coordinate):
    strLP = 'L{part}P'.format(part=part)
    strRP = 'R{part}P'.format(part=part)
    l = np.sum(df[str('{str}'.format(str=strLP))]) / totalFrames
    r = np.sum(df[str('{str}'.format(str=strRP))]) / totalFrames
    if coordinate == 'y' or coordinate == 'Y':
        if l > r:
            return 'L{part}Y'.format(part=part)
        else:
            return 'R{part}Y'.format(part=part)
    elif coordinate == 'x' or coordinate == 'X':
        if l > r:
            return 'L{part}X'.format(part=part)
        else:
            return 'R{part}X'.format(part=part)


# variables are left or right depending which has the highest probability over the dataset
wristY = largest('wrist', 'Y')
shoulderY = largest('shoulder', 'Y')
elbowY = largest('elbow', 'Y')
kneeY = largest('knee', 'Y')

wristX = largest('wrist', 'X')
shoulderX = largest('shoulder', 'X')
elbowX = largest('elbow', 'X')
kneeX = largest('knee', 'X')

# RATE OF CHANGE shoulder
n = 5
# Y
ROCYshoulder = df['{part}'.format(part=shoulderY)]
ROCY = ROCYshoulder.pct_change(periods=-n)
ROCY = (ROCY.abs()) * 100

# X
ROCXshoulder = df['{part}'.format(part=shoulderX)]
ROCX = ROCXshoulder.pct_change(periods=-n)
ROCX = (ROCX.abs()) * 100

framecolumn = df['framenum']
dfROCY = pd.concat([ROCY, framecolumn], axis=1)
dfROCY.columns = ['{part}ROCY'.format(part=shoulderY), 'framenum']

dfROCX = pd.concat([ROCX, framecolumn], axis=1)
dfROCX.columns = ['{part}ROCX'.format(part=shoulderX), 'framenum']


dfROCSUM = ROCY.add(ROCX, fill_value=0)
dfROCSUM = pd.concat([dfROCSUM, framecolumn], axis=1)
dfROCSUM.columns = ['totalROC','framenum']

ax1 = dfROCSUM.plot(kind='line',x='framenum',y='totalROC', color='r')
plt.show()