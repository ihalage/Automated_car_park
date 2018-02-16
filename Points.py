#!/usr/bin/env python

import time
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

img = cv2.imread('avgbackground.png', 1)
Emptyimg = img.copy()
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

emptyavg = np.mean(imgray[250:300,400:470])

#img[250:300,400:470] = (0, 255, 0)

cap = cv2.VideoCapture('cam1.mp4')

while(cap.isOpened()):
	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	
	vehavg = np.mean(gray[250:300,400:470])

	if abs(vehavg-emptyavg)>30:
		img[250:300,400:470] = (0, 0, 255)
	else:
		img[250:300,400:470] = (0, 255, 0)

	cv2.imshow('park',frame)
	cv2.imshow('empty',img)
	time.sleep(0.01)

	#cv2.imwrite('croppedimage.png', frame)
	#break

cap.release()
cv2.destroyAllWindows()

#Pimg = cv2.imread('avgvehicle.png', 1)
#PEmptyimg = Pimg.copy()
#imgParkedgray = cv2.cvtColor(Pimg,cv2.COLOR_BGR2GRAY)

#Pimg[250:300,400:470] = (0, 255, 0)

#vehavg = np.mean(imgParkedgray[250:300,400:470])

#print emptyavg
#print vehavg

#cv2.imshow('image',img)
#cv2.imshow('thresh',imgParkedgray)
#cv2.imshow('imgrayneg',imgrayNeg)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#TL 400 250
#TR 480 250
#BL 390 300
#BR 470 310
