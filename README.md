# DeadliftPoseCorrection
## Prototypes using OpenPose, to estimate poses, identify checkpoints, classify postures and provide feedback

### Must ensure that OpenPose is downloaded and installed on a MAC to work with the OpenPose Python API.
### Other libraries include Pytorch, OpenCV, Numpy, Pandas, Matplotlib, Math, sklearn.

**video2csv.py** --- Extract keypoints from a video and convert to a csv file. 

**csv2checkpoints.py** --- Identifies checkpoints using rate of change and y key point intersections from the csv file and outputs example images .

**scalePostures.py** --- Functionality to scale and align postures to a specific point and size.

**ANN.py** --- Pytorch neural netork training/testing model.

**plotSkeletons.py** --- visual guidance from reference posture and input posture, aswell as bearing and angle functionality.

**plotcsv.py** --- plots frame by frame analysis for a visualisation of key point intersections.

**plotData.py** --- plots a percentage of a data set to see the visual differences between class key points.

**ROCcsv.py** --- Plots rate of change for specific key points frame by frame.

**mergeCSV.py** --- scales and aligns the 3 class postures used for each checkpoint.

**dataSynthesis.py** --- generates synthetic data from existing datasets.

**splitData.py** --- splits dataset into training and testing sets.

