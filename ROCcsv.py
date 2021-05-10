import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#change to desired filepath for csv file
df = pd.read_csv('badOutputs/CSV/outputBAD2.csv')
df = df.replace(0, np.nan)

#RATE OF CHANGE for SHOULDER
n =4 #rate of change calculated every 4 frames rather than every 1
#Y
RshoulderY = df['RshoulderY']
ROCY= RshoulderY.pct_change(periods=-n)
ROCY= (ROCY.abs())*100 # makes it a percentage
#X
RshoulderX = df['RshoulderX']
ROCX = RshoulderX.pct_change(periods=-n)
ROCX = (ROCX.abs())*100


framecolumn = df['framenum']
#Y
dfROCY = pd.concat([ROCY, framecolumn], axis =1)
dfROCY.columns =['RshoulderROCY', 'framenum']
#X
dfROCX = pd.concat([ROCX, framecolumn], axis = 1)
dfROCX.columns =['RshoulderROCX', 'framenum']

#sum X and Y ROC for total
dfROCSUM = ROCY.add(ROCX, fill_value=0)
dfROCSUM = pd.concat([dfROCSUM, framecolumn], axis=1)
dfROCSUM.columns = ['totalROC','framenum']

#plot Rate of change for the shoulders X and Y values
ax1 = dfROCY.plot(kind='line',x='framenum',y='RshoulderROCY', color='r')
ax2 = dfROCX.plot(kind='line', x='framenum', y='RshoulderROCX', color='blue', ax=ax1)
ax3 = dfROCSUM.plot(kind='line', x='framenum', y='totalROC', color='black')
ax1.set_ylabel("Rate of Change %: R shoulder")
plt.show()

