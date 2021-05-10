import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#using frame data from openpose body 25(more accurate)
#rename to desired file path to csv
df = pd.read_csv('goodOutputs/CSV/ytgood1.csv')
df = df.replace(0, np.nan)


ax1= df.plot(kind='scatter',x='framenum',y='RwristY', color='r') # scatter plot
#ax2= df.plot(kind='scatter',x='framenum',y='LwristY', color='r', ax=ax1) # scatter plot
ax3= df.plot(kind='scatter',x='framenum',y='RkneeY', color='y', ax=ax1)
#ax4= df.plot(kind='scatter',x='framenum',y='LkneeY', color='y', ax=ax1)
ax5= df.plot(kind='scatter',x='framenum',y='RelbowY', color='black', ax=ax1)
#ax6= df.plot(kind='scatter',x='framenum',y='LelbowY', color='black', ax=ax1)
ax7= df.plot(kind='scatter',x='framenum',y='midhipY', color='pink', ax=ax1)
#ax8= df.plot(kind='scatter',x='framenum',y='LshoulderY', color='grey', ax=ax1)
ax9= df.plot(kind='scatter',x='framenum',y='RshoulderY', color='grey', ax=ax1)
# Plot legend.
#plt.legend(['Rwrist','Lwrist', 'Rknee','Lknee', 'Relbow','Lelbow', 'Midhip', 'Rshoulder', 'Lshoulder'])
plt.legend(['Rwrist','Rknee','Relbow','Midhip', 'Rshoulder'])
plt.show()
