import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#using frame data from openpose body 25
#Rename df below to desired csv file path
df = pd.read_csv('goodOutputs/CSV/outputGOOD1(0).csv')
df = df.replace(0, np.nan)

totalFrames = df['framenum'].max()
wristsMax = np.sum(df[['LwristY','RwristY']].max())/2
shouldersMax = np.sum(df[['LshoulderY', 'RshoulderY']].max())/2
elbowsMax = np.sum(df[['LelbowY', 'RelbowY']].max())/2
hipMax = df[['midhipY']].max()


#calculates and returns which side rY or lY has the highest probability for a known part
def largest(part, coordinate):
    strLP = 'L{part}P'.format(part=part)
    strRP = 'R{part}P'.format(part=part)
    l = np.sum(df[str('{str}'.format(str=strLP))])/totalFrames
    r = np.sum(df[str('{str}'.format(str=strRP))])/totalFrames
    if coordinate == 'y' or coordinate == 'Y':
        if l > r:
            return 'L{part}Y'.format(part=part)
        else:
            return 'R{part}Y'.format(part=part)
    elif coordinate =='x' or coordinate == 'X':
        if l > r:
            return 'L{part}X'.format(part=part)
        else:
            return 'R{part}X'.format(part=part)



#variables are left or right depending which has the highest probability over the dataset
wristY = largest('wrist','Y')
shoulderY = largest('shoulder','Y')
elbowY = largest('elbow','Y')
kneeY = largest('knee','Y')

wristX = largest('wrist','X')
shoulderX = largest('shoulder','X')
elbowX = largest('elbow','X')
kneeX = largest('knee','X')

#RATE OF CHANGE shoulder
n =4
#Y
ROCYshoulder = df['{part}'.format(part=shoulderY)]
ROCY= ROCYshoulder.pct_change(periods=-n)
ROCY= (ROCY.abs())*100

#X
ROCXshoulder = df['{part}'.format(part=shoulderX)]
ROCX= ROCXshoulder.pct_change(periods=-n)
ROCX= (ROCX.abs())*100

framecolumn = df['framenum']
dfROCY = pd.concat([ROCY, framecolumn], axis =1)
dfROCY.columns =['{part}ROCY'.format(part=shoulderY), 'framenum']

dfROCX = pd.concat([ROCX, framecolumn], axis =1)
dfROCX.columns =['{part}ROCX'.format(part=shoulderX), 'framenum']

#sum X and Y ROC for total
dfROCSUM = ROCY.add(ROCX, fill_value=0)
dfROCSUM = pd.concat([dfROCSUM, framecolumn], axis=1)
dfROCSUM.columns = ['totalROC','framenum']

# OLD!!!!!!! Checkpoint 1 - wrists, shoulders and elbows all leave their local maximum
# c1querystring = '{part1} <= @wristsMax and {part2} <= @shouldersMax and {part3} <= @elbowsMax'.format(part1=wrist,part2=shoulder,part3=elbow)
# c1query = df.query(c1querystring)
# c1frame = c1query['framenum'].min()
# c1row = df.query('framenum == @c1frame')

#Checkpoint 1 ROC earliest frame where ROC % is greater than a threshold
threshold = 1.0
c1querystring = 'totalROC >= @threshold'
c1query = dfROCSUM.query(c1querystring)
c1frame = c1query['framenum'].min()
c1row = df.query('framenum == @c1frame')

#Checkpoint 2 - wrists intersect knees
c2querystring='{part1} <= {part2} and framenum > @c1frame'.format(part1=wristY,part2=kneeY)
c2query = df.query(c2querystring)
c2frame = c2query['framenum'].min()
c2row = df.query('framenum == @c2frame')


#Checkpoint 3 - elbows intersect mid hip
c3querystring='{part1} <= midhipY and framenum > @c2frame'.format(part1=elbowY)
c3query = df.query(c3querystring)
c3frame = c3query['framenum'].min()
c3row = df.query('framenum == @c3frame')


#Checkpoint 4 - earliest frame number where ROC is below of equal a threshold
c4querystring= 'totalROC <= @threshold and framenum > @c3frame'
c4query = dfROCSUM.query(c4querystring)
c4frame = c4query['framenum'].min()
c4row = df.query('framenum == @c4frame')


#Checkpoint 5 - elbows intersect mid hip going down
c5querystring = '{part1} <= midhipY and framenum > @c4frame'.format(part1=elbowY)
c5query = df.query(c5querystring)
c5frame = c5query['framenum'].max()
c5row = df.query('framenum == @c5frame')

#
#Checkpoint 6 - wrists intersect knee doing down
c6querystring = '{part1} <= {part2} and framenum > @c5frame'.format(part1=wristY, part2=kneeY)
c6query = df.query(c6querystring)
c6frame = c6query['framenum'].max()
c6row = df.query('framenum == @c6frame')


# # OLD!!! Checkpoint 7 - finds the frame with maximum Y after the c6frame row
# c7querydata= df.query('framenum > @c6frame')
# c7query = c7querydata[wrist].max()
# c7querystring ='{part1} == @c7query'.format(part1=wrist)
# c7row = c7querydata.query(c7querystring)
# c7frame = int(c7row['framenum'])

#Checkpoint 7 earliest frame after c6 where ROC % is lower than a threshold
c7querystring= 'totalROC <= @threshold and framenum > @c6frame'
c7query = dfROCSUM.query(c7querystring)
c7frame = c7query['framenum'].min()
c7row = df.query('framenum == @c7frame')

#show images along with checkpoint number in same window
frames = [c1frame,c2frame,c3frame,c4frame,c5frame,c6frame,c7frame]

plt.figure(figsize=(10,1))


for i in range(len(frames)):
    n = i+1
    #rename to path for relevant frame images
    f = 'goodOutputs/FRAMES/opframesGOOD1/frame{num}.jpg'.format(num=frames[i])
    try:
        img = plt.imread(f)
        plt.subplot(1, 7, i + 1).text(0.5,-0.1, "Checkpoint {n}".format(n=n), size=8)
        plt.imshow(img)
        plt.axis('off')
    except:
        print("There is a NaN")
print(frames)

plt.show()


