import cv2
import numpy as np
import glob
import re

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [
        int(text)
        if text.isdigit() else text.lower()
        for text in _nsre.split(s)]
images = []

for jpg in glob.glob('/Users/david-webber/openpose/build/examples/tutorial_api_python/goodOutputs/FRAMES/opframes_ytgood2/*.jpg'):
    images.append(jpg)
#sorted paths list to iterate through to get ordered video
sorted_paths = sorted(images, key=natural_sort_key)

img_array = []
for filename in sorted_paths:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

#output goes to tutorials folder!!!
out = cv2.VideoWriter('project.mp4', 0x7634706d, 15, size)
#
for i in range(len(img_array)):

    out.write(img_array[i])
out.release()