import cv2
import matplotlib.pyplot as plt
import numpy as np
cap = cv2.VideoCapture("../../../examples/media/sideviewBAD2.mp4")
cap.set(1, 10-1)
res, frame = cap.read()
edges = cv2.Canny(frame, 500,500)
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
vectorized = img.reshape((-1,3))
vectorized = np.float32(vectorized)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

res = center[label.flatten()]
result_image = res.reshape(img.shape)
figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.show()

# cv2.imshow('window_name', img)
# while True:
#     ch = 0xFF & cv2.waitKey(1) # Wait for a second
#     if ch == 27:
#         break