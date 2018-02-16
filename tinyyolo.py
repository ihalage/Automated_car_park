from darkflow.net.build import TFNet
import cv2

# initializing yolo
options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "tiny-yolo-voc.weights", "threshold": 0.6, "gpu":1.0}  # 
tfnet = TFNet(options)
img = cv2.imread('track/im_6.png')
yoloresult = tfnet.return_predict(img)
print yoloresult
for detectObj in yoloresult:
	cv2.rectangle(img, (detectObj['topleft']['x'],detectObj['topleft']['y']), (detectObj['bottomright']['x'],detectObj['bottomright']['y']), (255, 255, 255), 3)
cv2.imwrite('track.png', img)
# cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
# while True:
# 	cv2.imshow("Image", img)
