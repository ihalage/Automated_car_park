import cv2
import numpy as np

img1 = cv2.imread('referance.png',0)
#img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#img2 = cv2.imread('star2.jpg',0)

ret, thresh = cv2.threshold(img1, 255, 0,255)
#ret, thresh2 = cv2.threshold(img2, 127, 255,0)
img1, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt1 = contours[0]
#contours,hierarchy = cv2.findContours(thresh2,2,1)
#cnt2 = contours[0]

#ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
#print ret

cv2.drawContours(img1, contours, -1, (0,255,0), 3)
cv2.imshow("egvewqf", img1)

while True:
	if cv2.waitKey(1) & 0xff== ord('q'):
                break
