import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

#JOINTS
# //     {0,  "Nose"},
# //     {1,  "Neck"},
# //     {2,  "RShoulder"},
# //     {3,  "RElbow"},
# //     {4,  "RWrist"},
# //     {5,  "LShoulder"},
# //     {6,  "LElbow"},
# //     {7,  "LWrist"},
# //     {8,  "MidHip"},
# //     {9,  "RHip"},
# //     {10, "RKnee"},
# //     {11, "RAnkle"},
# //     {12, "LHip"},
# //     {13, "LKnee"},
# //     {14, "LAnkle"},
# //     {15, "REye"},
# //     {16, "LEye"},
# //     {17, "REar"},
# //     {18, "LEar"},
# //     {19, "LBigToe"},
# //     {20, "LSmallToe"},
# //     {21, "LHeel"},
# //     {22, "RBigToe"},
# //     {23, "RSmallToe"},
# //     {24, "RHeel"},
# //     {25, "Background"}

#function to turn row into workable list for plotting
def processRow(row):
    list = []
    for i in range(len(row[0])):
        list.append(row[0][i])
    #deletes every third element in list(probability)
    del list[2::3]
    #deletes last element (frame number)
    del list[-1]
    #gets the x and y coordinates and zip them into tuples to be used for plotting
    x = list[::2]
    y = list[1::2]
    coordinates = zip(x,y)
    return coordinates

#using frame data from reference
df = pd.read_csv('goodOutputs/CSV/outputGOOD1(0).csv')
df = df.replace(0, np.nan)

#c4 reference 37
row= df.query('framenum == 37').values.tolist()


#using frame data from input
dfin = pd.read_csv('badOutputs/CSV/outputBAD2.csv')
dfin = dfin.replace(0, np.nan)

#c4 input 50
rowin= dfin.query('framenum == 50').values.tolist()

coordinatesREF = list(processRow(row))
coordinatesIN = list(processRow(rowin))

x_valREF = [x[0] for x in coordinatesREF]
y_valREF = [x[1] for x in coordinatesREF]
x_valIN = [x[0] for x in coordinatesIN]
y_valIN = [x[1] for x in coordinatesIN]

#body length between neck keypoint and midhip, useful for normalisation
bodylengthREF = math.hypot(coordinatesREF[1][0]-coordinatesREF[8][0], coordinatesREF[1][1]-coordinatesREF[8][1])
bodylengthIN = math.hypot(coordinatesIN[1][0]-coordinatesIN[8][0], coordinatesIN[1][1]-coordinatesIN[8][1])
#this is used to scale reference body part lengths with input lengths
#by multiplying differencePCT with reference (x,y points)
differencePCT = bodylengthIN/bodylengthREF

#scales reference with inputs torso length
scaledValREFX, scaledValREFY = [x * differencePCT for x in x_valREF], [y * differencePCT for y in y_valREF]

#aligns scaled reference with inputs frame, using midhip [8] has center point
midhipINX,midhipINY = x_valIN[8], y_valIN[8]
midhipREFX,midhipREFY = scaledValREFX[8],scaledValREFY[8]

x_diff = midhipINX - midhipREFX
y_diff = midhipINY - midhipREFY


alignedREFX, alignedREFY = [x + x_diff for x in scaledValREFX],[y + y_diff for y in scaledValREFY]

point_pairs = [[1, 0], [1, 2], [1, 5],
                [2, 3], [3, 4], [5, 6],
                [6, 7], [0, 15], [15, 17],
                [0, 16], [16, 18], [1, 8],
                [8, 9], [9, 10], [10, 11],
                [11, 22], [22, 23], [11, 24],
                [8, 12], [12, 13], [13, 14],
                [14, 19], [19, 20], [14, 21]]

# plot lines between keypoints (draws skeleton)
for pair in point_pairs:
    partA = pair[0]
    partB = pair[1]
    if math.isnan(coordinatesIN[partA][0])or math.isnan(coordinatesIN[partB][0]):
        continue
    else:
        x_values =[coordinatesIN[partA][0], coordinatesIN[partB][0]]
        y_values =[coordinatesIN[partA][1], coordinatesIN[partB][1]]
        plt.plot(x_values, y_values, 'red')


for pair in point_pairs:
    partA = pair[0]
    partB = pair[1]
    x_values = [alignedREFX[partA], alignedREFX[partB]]
    y_values = [alignedREFY[partA], alignedREFY[partB]]
    plt.plot(x_values, y_values, 'green')


#plt.scatter(x_valREF,y_valREF)
inputPose = plt.scatter(x_valIN, y_valIN, label='Input Posture',c='red')
referencePose = plt.scatter(alignedREFX, alignedREFY, label='Reference Posture',c='green')
plt.title("Reference Checkpoint Pose vs Input Checkpoint Pose")
plt.legend(handles =[inputPose,referencePose], loc='lower right')

#flips the plot
plt.gca().invert_yaxis()
# #sets x and y axis equal so no skewing
plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')
plt.show()


##TODO: angles and determine what angles to calculate for each checkpoint
# coordinates x,y
# center point = midhip, other points = Rshoulder and Rknee
#angle = math.atan2(shoulder y - midhip y, shoulder x - midhip x) - math.atan2(knee y - midhip y, knee x - midhip x)

angleTorsoIN = math.atan2(y_valIN[2] - y_valIN[8], x_valIN[2]-x_valIN[8]) - math.atan2(y_valIN[10] - y_valIN[8],x_valIN[10] - x_valIN[8])
val = 180/math.pi
angleTorsoREF = math.atan2(y_valREF[2] - y_valREF[8], x_valREF[2]-x_valREF[8]) - math.atan2(y_valREF[10] - y_valREF[8],x_valREF[10] - x_valREF[8])

# print("Input Torsos angle:")
# print(abs(angleTorsoIN * val))
# print("Reference Torsos angle:")
# print(abs(angleTorsoREF * val))

def calculateBearing(pivot_x, pivot_y, x, y):
    angle = math.degrees(math.atan2(y - pivot_y, x - pivot_x))
    bearing = (90 - angle) % 360
    diff = bearing-180
    flip = ((180 -diff)+180)%360
    return flip

#djrection of hip to shoulder INPUT
INhipshoulderB= calculateBearing(x_valIN[8], y_valIN[8], x_valIN[2],y_valIN[2])
#direction of elbow to wrists INPUT
INelbowwristB= calculateBearing(x_valIN[3],y_valIN[3],x_valIN[4],y_valIN[4])
#direction of hip to knee INPUT
INhipkneeB= calculateBearing( x_valIN[8],y_valIN[8],x_valIN[10], y_valIN[10])
#direction of shoulder to elbow INPUT
INshoulderelbowB= calculateBearing(x_valIN[2],y_valIN[2],x_valIN[3],y_valIN[3])
#direction of knee to ankle INPUT
INkneeankleB=calculateBearing(x_valIN[10], y_valIN[10],x_valIN[11], y_valIN[11])

#djrection of hip to shoulder REF
REFhipshoulderB= calculateBearing(alignedREFX[8], alignedREFY[8], alignedREFX[2],alignedREFY[2])
#direction of elbow to wrists REF
REFelbowwristB= calculateBearing(alignedREFX[3],alignedREFY[3],alignedREFX[4],alignedREFY[4])
#direction of hip to knee REF
REFhipkneeB= calculateBearing( alignedREFX[8],alignedREFY[8],alignedREFX[10], alignedREFY[10])
#direction of shoulder to elbow REF
REFshoulderelbowB= calculateBearing(alignedREFX[2],alignedREFY[2],alignedREFX[3],alignedREFY[3])
#direction of knee to ankle REF
REFkneeankleB=calculateBearing(alignedREFX[10], alignedREFY[10],alignedREFX[11], alignedREFY[11])

def circularRange(start, end, modulo):
    start = round(start)
    end = round(end)
    if start > end:
        while start < modulo:
            yield start
            start += 1
        start = 0

    while start < end:
        yield start
        start += 1
def degreesDistance(start, end):
    diff =(end - start) % 360
    sDistance = 180 - abs((abs(diff)-180))
    return sDistance


def IN2REFbearing(inputb,refb, threshold, part):
    inputb = round(inputb)
    refb = round(refb)
    #a holds the total number of points that arent in the range midpoint == a/2
    a = 360 - (threshold*2)
    #finds mid-point that isnt the range
    bminus= abs(((refb-threshold)%360) -(a/2))
    bplus = abs(((refb+threshold)%360) -(a/2))

    if bminus < bplus:
        closestpoint = (refb-threshold)%360
    else:
        closestpoint = (refb+threshold)%360

    if inputb in circularRange((refb-threshold)%360,(refb+threshold)%360, 360):
        print('Good form!')
    else:
        print('Bad form!')
        print('Reference bearing is: ' +str(refb))
        print('Input bearing is: '+ str(inputb))
        #test this see if it works properly

        if (closestpoint - inputb + 360) %360 < 180:
            print('Move {part} clockwise: '.format(part=part)+str(degreesDistance(inputb,refb))+' Degrees')
        else:
            print('Move {part} anti-clockwise: '.format(part=part)+str(degreesDistance(inputb,refb))+' Degrees')


IN2REFbearing(INhipshoulderB, REFhipshoulderB, 5, 'torso')





