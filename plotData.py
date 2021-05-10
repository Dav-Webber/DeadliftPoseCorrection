import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


df = pd.read_csv('d-set-full.csv')
c0 = df.loc[df['class'] == 0] #straight
c1 = df.loc[df['class'] == 1] #back
c2 = df.loc[df['class'] == 2] #forward

c0 = c0.drop(columns=['class'])
c1 = c1.drop(columns=['class'])
c2 = c2.drop(columns=['class'])

fig = plt.figure()
ax1 = fig.add_subplot(111)

for i in range(round(c0.shape[0]/2)):
    for j in range(0,12,2):

        ax1.scatter(x=c0.iloc[i][j],y=c0.iloc[i][j+1], c='red')
for i in range(round(c1.shape[0]/2)):
    for j in range(0,12,2):
        ax1.scatter(x=c1.iloc[i][j],y=c1.iloc[i][j+1], c='green')

for i in range(round(c2.shape[0]/2)):
    for j in range(0,12,2):
        ax1.scatter(x=c2.iloc[i][j],y=c2.iloc[i][j+1], c='blue')

plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')
red_patch = mpatches.Patch(color='red', label='Correct posture')
green_patch = mpatches.Patch(color='green', label='Leaning too far back')
blue_patch = mpatches.Patch(color='blue', label='Leaning too far forward')
plt.legend(handles=[red_patch, green_patch, blue_patch])
plt.show()