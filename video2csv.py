import sys
import cv2
import os
from sys import platform
import matplotlib as plt
import numpy as np
import csv


def listrow2csv(csvname, input):
    with open("{fname}.csv".format(fname=csvname), "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(input)


try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    csvHeader = ['noseX', 'noseY', 'noseP', 'neckX', 'neckY', 'neckP',
                 'RshoulderX', 'RshoulderY', 'RshoulderP', 'RelbowX', 'RelbowY', 'RelbowP',
                 'RwristX', 'RwristY', 'RwristP', 'LshoulderX', 'LshoulderY', 'LshoulderP',
                 'LelbowX', 'LelbowY', 'LelbowP', 'LwristX', 'LwristY', 'LwristP',
                 'midhipX', 'midhipY', 'midhipP', 'RhipX', 'RhipY', 'RhipP',
                 'RkneeX', 'RkneeY', 'RkneeP', 'RankleX', 'RankleY', 'RankleP',
                 'LhipX', 'LhipY', 'LhipP', 'LkneeX', 'LkneeY', 'LkneeP',
                 'LankleX', 'LankleY', 'LankleP', 'ReyeX', 'ReyeY', 'ReyeP',
                 'LeyeX', 'LeyeY', 'LeyeP', 'RearX', 'RearY', 'RearP',
                 'LearX', 'LearY', 'LearP', 'LbigtoeX', 'LbigtoeY', 'LbigtoeP',
                 'LsmalltoeX', 'LsmalltoeY', 'LsmalltoeP', 'LheelX', 'LheelY', 'LheelP',
                 'RbigtoeX', 'RbigtoeY', 'RbigtoeP', 'RsmalltoeX', 'RsmalltoeY', 'RsmalltoeP',
                 'RheelX', 'RheelY', 'RheelP', 'framenum']

    # creates a csv if it doesnt exist and writes the headers from above
    # rename this file to relevant csv name for this video
    with open("c3-forward-full.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(csvHeader)

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"
    # params["number_people_max"] = 1
    params["disable_blending"] = True  # for black background
    params["display"] = 0

    # Starting OpenPose #
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    # rename the video file name and path for video you want
    cap = cv2.VideoCapture("../../../examples/media/c3-forward-27.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    framecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Total frames in this video: ' + str(framecount))
    framenum = 0

    # rename this directory to relevant directory name
    directory = 'c3-forward-full'
    if not os.path.exists(directory):
        os.makedirs(directory)

    while (cap.isOpened()):
        framenum += 1
        hasFrame, frame = cap.read()
        if hasFrame == True:
            datum.cvInputData = frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            opframe = datum.cvOutputData
            height, width, layers = opframe.shape

            cv2.imwrite(directory + "/frame%d.jpg" % framenum, opframe)

            # make list to be new row for a csv file
            a = []
            for i in range(25):
                for j in range(3):
                    a.append(datum.poseKeypoints[0][i][j].tolist())
            a.append(framenum)
            listrow2csv(directory, a)
            print(str(framenum) + ' out of a total ' + str(framecount) + ' frames have been pose estimated.')
        else:
            break

except Exception as e:
    print(e)

    sys.exit(-1)
