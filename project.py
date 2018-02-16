# Import the required modules
import dlib
from darkflow.net.build import TFNet
import cv2
import argparse as ap
import get_points
import json
import time
import firebase
import numpy as np
import sys
import caffe
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import math
import pylab
import csv
import collections
import itertools
import tensorflow as tf
import functions
import deep_net
from PIL import Image

def detect(im, param_vals):
    """
    Detect number plates in an image.

    :param im:
            Image to detect number plates in.

    :param param_vals:
            Model parameters to use. These are the parameters output by the `train`
            module.

    :returns:
            a 7,36 matrix giving the probability distributions of each letter.

    """

    # Load the model which detects number plates
    x, y, params = deep_net.final_training_model()

    # Execute the model at each scale.
    with tf.Session(config=tf.ConfigProto()) as sess:
        feed_dict = {x: np.stack([im])}
        feed_dict.update(dict(zip(params, param_vals)))
        y_val = sess.run(y, feed_dict=feed_dict)

    #finding the probabilities of each letter being present
    letter_probs = y_val.reshape(7,len(functions.CHARS))
    letter_probs = functions.softmax(letter_probs)

    return letter_probs

#Joining the letters with maximum probability
def letter_probs_to_code(letter_probs):
    return "".join(functions.CHARS[i] for i in np.argmax(letter_probs, axis=1))


def run(source=0, source2=0, source3=0):
    # ---- Initializing DNN  -----
    from sklearn.preprocessing import normalize
    caffe.set_mode_gpu()
    # cap=cv2.VideoCapture('../../vd1.mp4')
   #net = caffe.Net('../../segnet_inference.prototxt','../../segnet_iter_10000.caffemodel',caffe.TEST)
    #net = caffe.Net('../../enet_deploy_final.prototxt','../../segnet_iter_32000.caffemodel',caffe.TEST)
    net = caffe.Net('../../enet_deploy_final.prototxt','../../segnet_iter_55000.caffemodel',caffe.TEST)
    
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    background = [255,0,0]
    car = [0,255,0]
    numberplate = [0,0,255]
    label_colours = np.array([background, car, numberplate])
    label_colour = np.array([2, 1, 0])
    cropped=[]
    resizeimg=[]
    # ---- End of Initializing DNN  -----
    # ---- Initializing Tracker  -----
    # initializing yolo
    options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "tiny-yolo-voc.weights", "threshold": 0.5}  # , "gpu":1.0
    tfnet = TFNet(options)

    #----violation detection intial data starts----
    temp_count = 0
    pre_plate=''
    URL = "fypandroid-gpu"
    slot1_violation = "no"
    slot2_violation = "no"
    slot1_occupancy = "free"
    slot2_occupancy = "free"
    slot1_violation_new = "no"
    slot2_violation_new = "no"
    slot1_occupancy_new = "free"
    slot2_occupancy_new = "free"
    update_na = 0
    slot1_num = "n/a"
    slot2_num = "n/a"
    updateonce = 0
    
    firebase.patch(URL, {'Slot_01_vehicle_number': "n/a"})
    firebase.patch(URL, {'Slot_02_vehicle_number': "n/a"})

    imgavgback = cv2.imread(avgbackgrndpath, 1)
    imgavgbackori = imgavgback.copy()
    imgavgbackorigray = cv2.cvtColor(imgavgback,cv2.COLOR_BGR2GRAY)
    imgavgback = imgavgback[80:400, 200:595]
    imgrayavgback = cv2.cvtColor(imgavgback,cv2.COLOR_BGR2GRAY)
    imgavgback -= 20

    #emptyavg1 = np.mean(imgavgbackorigray[100:180,400:470])
    #emptyavg2 = np.mean(imgavgbackorigray[205:300,400:470])

    MidPoints = []
    MidPoints.append([250,55]) #x,y slot 1
    MidPoints.append([255,170]) #x,y slot 2

    xyminmax = []
    xyminmax.append([135,390,5,120]) #xmin,xmax,ymin,ymax slot 1
    xyminmax.append([135,390,115,235]) #xmin,xmax,ymin,ymax slot 2
    #cap = cv2.VideoCapture('cam1.mp4')

    #----violation detection intial data ends----
    # Create the VideoCapture object
    cam = cv2.VideoCapture(source)
    cam2 = cv2.VideoCapture(source2)
    cap = cv2.VideoCapture(source3)
    # If Camera Device is not opened, exit the program
    if not cam.isOpened():
        print "Video device 1 or file 1 couldn't be opened"
        exit()
    if not cam2.isOpened():
        print "Video device 2 or file 2 couldn't be opened"
        exit()
    if not cap.isOpened():
        print "Video device 3 or file 3 couldn't be opened"
        exit()
    retval, img = cam.read()
    retval2, img2 = cam2.read()
    ret,frame = cap.read()
    if not retval:
        print "Cannot capture frame device 1"
        exit()
    if not retval2:
        print "Cannot capture frame device 2"
        exit()
    if not ret:
        print "Cannot capture frame device 3"
        exit()
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Image2", cv2.WINDOW_NORMAL)
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)
    cv2.imshow("Image2", img2)
    cv2.imshow("frame", frame)

    # values for tracking
    threshld = 10
    height = 480 -threshld
    width = 640 -threshld
    limit = height - 20
    tryframes = 8 

    points = []
    points2 = []

    # Create the tracker object
    tracker = [dlib.correlation_tracker() for _ in xrange(0)]
    tracker2 = [dlib.correlation_tracker() for _ in xrange(0)]
    # Provide the tracker the initial position of the object
    # [tracker[i].start_track(img, dlib.rectangle(*rect)) for i, rect in enumerate(points)]
    for k, rect in enumerate(points):
        tracker[k].start_track(img, dlib.rectangle(*rect))
    for k2, rect in enumerate(points2):
        tracker[k2].start_track(img2, dlib.rectangle(*rect))

    temp_tracker = list(tracker) #to remove the deleted trackers from tracker without affecting for loop 
    temp_tracker2 = list(tracker2)
    trigerYolo = 0
    justdetected = 100
    licenseplates = []
    licenseplates2 = []
    predetect = False
    # time.sleep(5)
    # ---- End Initializing Tracker  -----
    # count = 0  # facilitate to set the flag to start tracking (temporary) ****************
    # refimg=cv2.imread("111.jpg",0)
    # retref, refthresh = cv2.threshold(refimg, -1, 255,0)
    # refimg, refcontours, refhierarchy = cv2.findContours(refthresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # refcnt = refcontours[0]
    while True:
        # count += 1   # facilitate to set the flag to start tracking (temporary) ****************
        # print count
        # print "----- "+str(justdetected)+" --"
    # ---- DNN starts -----
        # time.sleep(.5)
        start=time.time()    
  
        # Read frame from device or file
        retval, img = cam.read()
        retval2, img2 = cam2.read()
        ret,frame=cap.read()
        img = cv2.resize(img, (640,480))
        img2 = cv2.resize(img2, (640,480))
        s=cv2.resize(frame,(480,360))
        img2ori = img2.copy()
        if not retval:
            print "Cannot capture frame device 1 | CODE TERMINATION....."
            exit()
        if not retval2:
            print "Cannot capture frame device 2 | CODE TERMINATION....."
            exit()
        if not ret:
            print "Cannot capture frame device 3 | CODE TERMINATION....."
            exit()

        justdetected += 1
        if justdetected >100 :
            justdetected =100 
        start_violation = 0
        start_tracking = False
        if justdetected > 4:
            
            #ret=cap.set(cv2.CAP_PROP_FPS,4)
            #ret=cap.set(3,360)
            #ret=cap.set(4,480)
            #frame = cv2.imread("myimage.png",1)
            transformer.set_transpose('data', (2,0,1))
            transformer.set_channel_swap('data', (2,1,0)) # if using RGB instead of BGR
            transformer.set_raw_scale('data', 255.0)
            net.blobs['data'].reshape(1,3,360,480)
            net.blobs['data'].data[...] = transformer.preprocess('data', s)

            output = net.forward()
            # print("---%s in sec"%(time.time()-start))
            predicted = net.blobs['prob'].data
            output = np.squeeze(predicted[0,:,:,:])
            ind = np.argmax(output, axis=0)
            p = ind.copy()
            d = ind.copy()
            r = ind.copy()
            g = ind.copy()
            b = ind.copy()
            for l in range(0,3):
                p[ind==l] = label_colour[l]
            cv2.imwrite("ind.png", p)
            ind = cv2.imread("ind.png",0)
            ctimage, contours, hierarchy = cv2.findContours(ind,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # Vimgc = cv2.drawContours(s, contours, -1, (0,0,255), 3)
            #label_colours = np.array([Sky, Building, Pole])
            for l in range(0,3):
                r[ind==l] = label_colours[l,0]
                g[ind==l] = label_colours[l,1]
                b[ind==l] = label_colours[l,2]
            # cv2.imshow('framex',d)
            rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
            rgb[:,:,0] = r/255.0
            rgb[:,:,1] = g/255.0
            rgb[:,:,2] = b/255.0
            #plt.figure()
            #plt.imshow(rgb,vmin=0, vmax=1)
            #plt.show()
             # im = cv2.imread(sys.argv[1])
             # im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.

            #f = numpy.load(sys.argv[2])
            #param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]

             # letter_probs = detect(im_gray, param_vals)


            #code = letter_probs_to_code(letter_probs)
             # print("Number Plate ->",code)
            #pre_plate='' 
            detectnow = False
            NoAllVContours = len(hierarchy[0])
            # print NoAllVContours 
            if(NoAllVContours<6):    
                for i in range(NoAllVContours):
                    M = cv2.moments(contours[i])
                    # print M['m00']
                    # print "#############################################################################################################################"
                    # print cv2.matchShapes(refcnt,contours[i],1,0.0)
                    if (5000>M['m00']>3000):
                        if (int(M['m01']/M['m00'])>170):
                            # if ( cv2.matchShapes(refcnt,contours[i],1,0.0) < 0.7):
                            x,y,w,h=cv2.boundingRect(contours[i])
                            cropped = s[y:y+h, x:x+w]
                            resizeimg=cv2.resize(cropped,(256,32))
                            cv2.imshow('framerty',resizeimg)
                            
                            im_gray = cv2.cvtColor(resizeimg, cv2.COLOR_BGR2GRAY) / 255.
                            start=time.time()
                            f = np.load("weights.npz")
                            param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]
                            letter_probs = detect(im_gray, param_vals)
                            code = letter_probs_to_code(letter_probs)
                            print("---%s in sec"%(time.time()-start))
                            #tempcode = code
                            if code[0] == "Z":
                                code = code[1:]
                            print("Number Plate ->",code)
                            detectnow = True
                            if (detectnow & predetect) : #((pre_plate == code) & (temp_count == 1)):
                                temp_count = 0
                                start_tracking = True
                                justdetected = 0
                                with open('GUI/VehicleNumber.csv','w') as f:
                                 writer=csv.writer(f)
                                 writer.writerow([code])
                            else:
                                if (pre_plate!=code):
                                    temp_count = 1
                                pre_plate=code
                            # elif(cv2.matchShapes(refcnt,contours[i],1,0.0)==0.0):   
                                # cv2.imshow('framex',rgb)     
                                # cv2.imwrite("sd.png", rgb)
                        # print pre_plate
            predetect = detectnow 
            # print licenseplates
            #print cap.get(cv2.CAP_PROP_FPS)
            #print frame.shape
            cv2.imshow('frame2',rgb)
            if cv2.waitKey(1) & 0xff== ord('q'):
                break
            # DNN sets the flag to start tracking
            # if count == 36:  # facilitate to set the flag to start tracking (temporary) ****************
            #     start_tracking = True
            #     print licenseplates
        # ---- DNN stops -----
        # ---- Tracker starts -----
        # print temp_tracker
        tracker = list(temp_tracker) 
        tracker2 = list(temp_tracker2) 
        
        
        new_points = []
        if trigerYolo > 0:  # if detection was not happened for the last frame
            yoloresult = tfnet.return_predict(img)
            for detectObj in yoloresult:
                if (inRegion(detectObj)):
                    vehicleloc = (detectObj['topleft']['x'],detectObj['topleft']['y']+20,detectObj['bottomright']['x'],detectObj['bottomright']['y']+10)
                    new_points.append(vehicleloc)
                    licenseplates.append(code)
                    print ("Tracking started for the request")
                    trigerYolo = 0
                    break
            print("Vehicle detection fails for the request for the next frame as well.")   
            trigerYolo -= 1
            cv2.imwrite('track/im_'+str(trigerYolo)+'.png', img)

        if(start_tracking):  #new vehicle trying to add if "p" pressed
            cv2.imwrite('track/im_'+str(trigerYolo)+'.png', img) # to save photos at tracking
            # yolo detection
            yoloresult = tfnet.return_predict(img) 
            for detectObj in yoloresult:
                if (inRegion(detectObj)):
                    vehicleloc = (detectObj['topleft']['x'],detectObj['topleft']['y']+20,detectObj['bottomright']['x'],detectObj['bottomright']['y']+10)
                    new_points.append(vehicleloc)
                    licenseplates.append(code)
                    # print licenseplates
                    print ("Tracking started for the request")
                    break # only one vehicle will be added per one request *************** optimize this to reduce issues with multi vehicles trying to enter at the same time
            if (len(new_points)<1):
                print "no vehicle found in frame 1: trying few other frames triggered"
                trigerYolo = tryframes   # trying few frames to capture a vehicle if not detected from first few frames

        
        if len(new_points) > 0: # if there is a new detected vehicled
            # Create the tracker object
            new_tracker = [dlib.correlation_tracker() for _ in xrange(len(new_points))]
            # Provide the tracker the new positions of the object
            for j, rect2 in enumerate(new_points):
                new_tracker[j].start_track(img, dlib.rectangle(*rect2))
            # [tracker[j].start_track(img, dlib.rectangle(*rect)) for j, rect in enumerate(points)]
            tracker.extend(new_tracker)
            print "Success: Tracking started for the newly entered vehicle."

        temp_tracker = list(tracker)
        # Update the tracker  
        for i in xrange(len(tracker)): #for number of objects to track
            confdnc = tracker[i].update(img)
            print confdnc
            
            # Get the position of th object, draw a 
            # bounding box around it and display it.
            rect = tracker[i].get_position()
            pt1 = (int(rect.left()), int(rect.top()))
            pt2 = (int(rect.right()), int(rect.bottom()))


            if (pt1[0]< threshld or pt1[1]< threshld or pt2[0]>=width):
                del licenseplates[i]
                del temp_tracker[i]
                print "Object removed: out of the frame/ unexpected behavior"
                continue
            if (pt2[1]>= limit):
                licenseplates2.append(licenseplates[i])
                del licenseplates[i]
                del temp_tracker[i]
                print "Object removed: vehicle enters next view"
                new_points2 = [(10, 150, 120, 260)]
                # new_tracker2 = dlib.correlation_tracker()
                # # Provide the tracker the new positions of the object
                # new_tracker2.start_track(img2, dlib.rectangle(*new_points2[0]))
                # # [tracker[j].start_track(img, dlib.rectangle(*rect)) for j, rect in enumerate(points)]
                # tracker2.append(new_tracker2)

                new_tracker2 = [dlib.correlation_tracker() for _ in xrange(len(new_points2))]
                # Provide the tracker the new positions of the object
                for j2, rect2_2 in enumerate(new_points2):
                    new_tracker2[j2].start_track(img2, dlib.rectangle(*rect2_2))
                # [tracker[j].start_track(img, dlib.rectangle(*rect)) for j, rect in enumerate(points)]
                tracker2.extend(new_tracker2)


                print "Success: Tracking started from view 2."
                continue
            # if (confdnc < 3.5):
            #     print "Low Confidence"
            #     # continue

            cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)
            # print "Object {} tracked at [{}, {}] \r".format(i, pt1, pt2),

            # show location of box if mentionedd
            loc = (int(rect.left()), int(rect.top()-20))
            txt = "Vehicle : [{}]".format(licenseplates[i])
            cv2.putText(img, txt, loc , cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
        temp_tracker2 = list(tracker2)
        for i2 in xrange(len(tracker2)): #for number of objects to track
            confdnc2 = tracker2[i2].update(img2)
            rect_2 = tracker2[i2].get_position()
            
            pt1_2 = (int(rect_2.left()), int(rect_2.top()))
            pt2_2 = (int(rect_2.right()), int(rect_2.bottom()))

            cv2.rectangle(img2, pt1_2, pt2_2, (255, 255, 255), 3)
            loc = (int(rect_2.left()), int(rect_2.top()-20))
            txt = "Vehicle : [{}]".format(licenseplates2[i2])
            cv2.putText(img2, txt, loc , cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
            if rect_2.right() >200:
                dist1 = ((MidPoints[0][0]- ((pt1_2[0]+pt2_2[0])/2))**2+(MidPoints[0][1]-((pt1_2[1]+pt2_2[1])/2))**2)
                dist2 = ((MidPoints[1][0]- ((pt1_2[0]+pt2_2[0])/2))**2+(MidPoints[1][1]-((pt1_2[1]+pt2_2[1])/2))**2)
                if dist1<dist2:
                    if slot1_num != txt:
                        firebase.patch(URL, {'Slot_01_vehicle_number': licenseplates2[i2]})
                        slot1_num = txt
                        if len(tracker2) == 1:
                            firebase.patch(URL, {'Slot_02_vehicle_number': "n/a"})
                else:
                    if slot2_num != txt:
                        firebase.patch(URL, {'Slot_02_vehicle_number': licenseplates2[i2]})
                        slot2_num = txt
                        if len(tracker2) == 1:
                            firebase.patch(URL, {'Slot_01_vehicle_number': "n/a"})
                start_violation = 1
                update_na = 1
            else:
                start_violation = 0
                if update_na == 1:
                    firebase.patch(URL, {'Slot_01_vehicle_number': "n/a"})
                    firebase.patch(URL, {'Slot_02_vehicle_number': "n/a"})
                    update_na = 0


        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)        # ***** commented *******
        # cv2.namedWindow("Image2", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
        cv2.imshow("Image2", img2)
        cv2.imshow('frame',s)
    # ---- end of tracker -----
    # ---- Violation detection starts ----
        #print "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq" + str(start_violation)
        if start_violation == 1:
            #retpark, carpark = cam2.read()
            carpark = img2ori
            # carpark = cv2.resize(carpark, (640,480))   # Width,Height
            #carparkori = carpark.copy()
            #carparkgrayori = cv2.cvtColor(carpark,cv2.COLOR_BGR2GRAY)

            #vehavg1 = np.mean(carparkgrayori[100:180,400:470])
            #vehavg2 = np.mean(carparkgrayori[205:300,400:470])
            updateonce = 0
            carpark = carpark[80:400, 200:595]
            carparkgray = cv2.cvtColor(carpark,cv2.COLOR_BGR2GRAY)

            #cv2.imshow('frame',frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break

            vehicles = cv2.subtract(carpark, imgavgback)
            #vehicles = carpark - imgavgback
            #vehiclesgray = imgavgbackgray - imgrayavgback
            vehiclesgray = cv2.cvtColor(vehicles,cv2.COLOR_BGR2GRAY)

            #retv,vehiclesgray = cv2.threshold(imgrayavgback,50,255,cv2.THRESH_BINARY)

            blur = cv2.GaussianBlur(vehiclesgray,(5,5),0)
            ret3,vehiclesgray = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            ret1,vehiclesgray = cv2.threshold(blur,ret3-25,255,cv2.THRESH_BINARY)

            #ret2,vehiclesgray = cv2.threshold(vehiclesgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            #vehiclesgray = cv2.adaptiveThreshold(vehiclesgray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
            #vehiclesgray = cv2.adaptiveThreshold(vehiclesgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)


            vcontimage, vehicontours, vehihierarchy = cv2.findContours(vehiclesgray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            NoAllContours = len(vehihierarchy[0])

            #print(NoAllContours)
            #print(vehicles)

            #Getting the parent Vcontour numbers
            VehicleNo = []

            #Mid points of vehicles
            VehicleMid = []

            for i in range(NoAllContours):
                #if (hierarchy[0,i,3]==-1):     
                    M = cv2.moments(vehicontours[i])
                    if ((M['m00']>3000)):
                            VehicleNo.append(i)
                            cx = int(M['m10']/M['m00'])
                            cy = int(M['m01']/M['m00'])
                            VehicleMid.append([cx,cy])


            #Number of all child countours
            NumOfVehicles = len(VehicleNo)

            #for i in range(NumOfVehicles):
                #Vimgc = cv2.drawContours(Pimg, contours, VehicleNo[i] , (0,0,255), 3)

            if (NumOfVehicles==0):
                    imgavgbackori[100:180,400:470] = (0, 255, 0)
                    imgavgbackori[205:300,400:470] = (0, 255, 0)
                    if slot1_occupancy != slot1_occupancy_new:
                        firebase.patch(URL, {'Slot_01_occupancy_status': 'free'})
                        slot1_occupancy = slot1_occupancy_new
                    if slot2_occupancy != slot2_occupancy_new:
                        firebase.patch(URL, {'Slot_02_occupancy_status': 'free'})
                        slot2_occupancy = slot2_occupancy_new
                    if slot1_violation != slot1_violation_new:
                        firebase.patch(URL, {'Slot_01_violations': 'no'})
                        slot1_violation = slot1_violation_new
                    if slot2_violation != slot2_violation_new:
                        firebase.patch(URL, {'Slot_02_violations': 'no'})
                        slot2_violation = slot2_violation_new
                    #if ((abs(vehavg1-emptyavg1))>30):
                        #img[100:180,400:470] = (0, 0, 255)
                    #else:
                        #img[100:180,400:470] = (0, 255, 0)
                    #if ((abs(vehavg2-emptyavg2))>30):
                        #img[205:300,400:470] = (0, 0, 255)
                    #else:
                        #img[205:300,400:470] = (0, 255, 0)
                #if (NumOfVehicles>0):
            else:
                    counter = 0
                    #error = 0

                    for i in VehicleNo:
                        NoIndices = len(vehicontours[i])
                        vehix = []  
                        vehiy = []
                        for j in range(NoIndices):
                                vehix.append(vehicontours[i][j][0,0])
                                vehiy.append(vehicontours[i][j][0,1])

                        #Vehicle Contour min max coordinate values
                        xmin = min(vehix)
                        xmax = max(vehix)
                        ymin = min(vehiy)
                        ymax = max(vehiy)

                        distance = []

                        for k in range(2):
                                vehid = ((MidPoints[k][0]-VehicleMid[counter][0])**2+(MidPoints[k][1]-VehicleMid[counter][1])**2)
                                distance.append(vehid)

                        parkingslot = distance.index(min(distance))

                        drawcon = 0
                        if (xmin < xyminmax[parkingslot][0]):
                                #error += 1
                                drawcon = 1
                        elif (xmax > xyminmax[parkingslot][1]):
                                #error += 1
                                drawcon = 1
                        elif (ymin < xyminmax[parkingslot][2]):
                                #error += 1
                                drawcon = 1
                        elif (ymax > xyminmax[parkingslot][3]):
                                #error += 1
                                drawcon = 1

                        if ((parkingslot == 0)&(drawcon == 1)):
                                veimgc = cv2.drawContours(carpark, vehicontours, i, (0,0,255), 3)
                                imgavgbackori[100:180,400:470] = (255, 0, 0)
                                slot1_violation_new = "yes"
                                slot1_occupancy_new = "occupied"
                                if slot1_occupancy != slot1_occupancy_new:
                                    firebase.patch(URL, {'Slot_01_occupancy_status': 'occupied'})
                                    slot1_occupancy = slot1_occupancy_new
                                if slot1_violation != slot1_violation_new:
                                    firebase.patch(URL, {'Slot_01_violations': 'yes'})
                                    slot1_violation = slot1_violation_new
                                if (NumOfVehicles == 1):
                                        imgavgbackori[205:300,400:470] = (0, 255, 0)
                                        slot2_violation_new = "no"
                                        slot2_occupancy_new = "free"
                                        if slot2_occupancy != slot2_occupancy_new:
                                            firebase.patch(URL, {'Slot_02_occupancy_status': 'free'})
                                            slot2_occupancy = slot2_occupancy_new
                                        if slot2_violation != slot2_violation_new:
                                            firebase.patch(URL, {'Slot_02_violations': 'no'})
                                            slot2_violation = slot2_violation_new
                        elif((parkingslot == 0)&(drawcon == 0)):
                                veimgc = cv2.drawContours(carpark, vehicontours, i, (0,255,0), 3)
                                imgavgbackori[100:180,400:470] = (0, 0, 255)
                                slot1_violation_new = "no"
                                slot1_occupancy_new = "occupied"
                                if slot1_occupancy != slot1_occupancy_new:
                                    firebase.patch(URL, {'Slot_01_occupancy_status': 'occupied'})
                                    slot1_occupancy = slot1_occupancy_new
                                if slot1_violation != slot1_violation_new:
                                    firebase.patch(URL, {'Slot_01_violations': 'no'})
                                    slot1_violation = slot1_violation_new
                                if (NumOfVehicles == 1):
                                        imgavgbackori[205:300,400:470] = (0, 255, 0)
                                        slot2_violation_new = "no"
                                        slot2_occupancy_new = "free"
                                        if slot2_occupancy != slot2_occupancy_new:
                                            firebase.patch(URL, {'Slot_02_occupancy_status': 'free'})
                                            slot2_occupancy = slot2_occupancy_new
                                        if slot2_violation != slot2_violation_new:
                                            firebase.patch(URL, {'Slot_02_violations': 'no'})
                                            slot2_violation = slot2_violation_new
                        elif ((parkingslot == 1)&(drawcon == 1)):
                                veimgc = cv2.drawContours(carpark, vehicontours, i, (0,0,255), 3)
                                imgavgbackori[205:300,400:470] = (255, 0, 0)
                                slot2_violation_new = "yes"
                                slot2_occupancy_new = "occupied"
                                if slot2_occupancy != slot2_occupancy_new:
                                    firebase.patch(URL, {'Slot_02_occupancy_status': 'occupied'})
                                    slot2_occupancy = slot2_occupancy_new
                                if slot2_violation != slot2_violation_new:
                                    firebase.patch(URL, {'Slot_02_violations': 'yes'})
                                    slot2_violation = slot2_violation_new
                                if (NumOfVehicles == 1):
                                        imgavgbackori[100:180,400:470] = (0, 255, 0)
                                        slot1_violation_new = "no"
                                        slot1_occupancy_new = "free"
                                        if slot1_occupancy != slot1_occupancy_new:
                                            firebase.patch(URL, {'Slot_01_occupancy_status': 'free'})
                                            slot1_occupancy = slot1_occupancy_new
                                        if slot1_violation != slot1_violation_new:
                                            firebase.patch(URL, {'Slot_01_violations': 'no'})
                                            slot1_violation = slot1_violation_new
                        elif((parkingslot == 1)&(drawcon == 0)):
                                veimgc = cv2.drawContours(carpark, vehicontours, i, (0,255,0), 3)
                                imgavgbackori[205:300,400:470] = (0, 0, 255)
                                slot2_violation_new = "no"
                                slot2_occupancy_new = "occupied"
                                if slot2_occupancy != slot2_occupancy_new:
                                    firebase.patch(URL, {'Slot_02_occupancy_status': 'occupied'})
                                    slot2_occupancy = slot2_occupancy_new
                                if slot2_violation != slot2_violation_new:
                                    firebase.patch(URL, {'Slot_02_violations': 'no'})
                                    slot2_violation = slot2_violation_new
                                if (NumOfVehicles == 1):
                                        imgavgbackori[100:180,400:470] = (0, 255, 0)
                                        slot1_violation_new = "no"
                                        slot1_occupancy_new = "free"
                                        if slot1_occupancy != slot1_occupancy_new:
                                            firebase.patch(URL, {'Slot_01_occupancy_status': 'free'})
                                            slot1_occupancy = slot1_occupancy_new
                                        if slot1_violation != slot1_violation_new:
                                            firebase.patch(URL, {'Slot_01_violations': 'no'})
                                            slot1_violation = slot1_violation_new

                        counter += 1
                    cv2.imshow('thresh',veimgc)
        else:
            imgavgbackori[100:180,400:470] = (0, 255, 0)
            imgavgbackori[205:300,400:470] = (0, 255, 0)
            if updateonce == 0:
            	firebase.patch(URL, {'Slot_01_occupancy_status': 'free'})
            	firebase.patch(URL, {'Slot_02_occupancy_status': 'free'})
            	firebase.patch(URL, {'Slot_01_violations': 'no'})
            	firebase.patch(URL, {'Slot_02_violations': 'no'})
            	slot1_violation_new = 'no'
            	slot2_violation_new = 'no'
            	slot1_occupancy_new = 'free'
            	slot2_occupancy_new = 'free'
            	slot1_occupancy = slot1_occupancy_new
            	slot2_occupancy = slot2_occupancy_new
            	slot1_violation = slot1_violation_new
            	slot2_violation = slot2_violation_new
            	updateonce = 1

            # if slot1_occupancy != slot1_occupancy_new:
            #     firebase.patch(URL, {'Slot_01_occupancy_status': 'free'})
            #     slot1_occupancy = slot1_occupancy_new
            # if slot2_occupancy != slot2_occupancy_new:
            #     firebase.patch(URL, {'Slot_02_occupancy_status': 'free'})
            #     slot2_occupancy = slot2_occupancy_new
            # if slot1_violation != slot1_violation_new:
            #     firebase.patch(URL, {'Slot_01_violations': 'no'})
            #     slot1_violation = slot1_violation_new
            # if slot2_violation != slot2_violation_new:
            #     firebase.patch(URL, {'Slot_02_violations': 'no'})

        cv2.imshow('image',imgavgbackori)
        #cv2.imshow('imgrayneg',vehiclesgray)
    # ---- Violation detection ends ----

    # ---- Empty parking space detection starts ----
        #ret, carpark = cam2.read()
        #carparkgray = cv2.cvtColor(carpark, cv2.COLOR_BGR2GRAY)

        ##cv2.imshow('frame',frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
    
        #vehavg = np.mean(carparkgray[250:300,400:470])

        #if abs(vehavg-emptyavg)>30:
            #imgavgback[250:300,400:470] = (0, 0, 255)
        #else:
            #imgavgback[250:300,400:470] = (0, 255, 0)

        #cv2.imshow('park',carpark)
        #cv2.imshow('empty',imgavgback)

    # ---- Empty parking space detection ends ----

    # Relase the VideoCapture object
    cam.release()
    cam2.release()
    cap.release()
    cv2.destroyAllwindows()

def inRegion(detectObj):
    if detectObj['label'] != 'car': return False
    if (detectObj['bottomright']['y'] > 80 and detectObj['topleft']['x'] > 130  and detectObj['bottomright']['x'] < 430 and detectObj['bottomright']['y'] > 130): return True  #
    return False 

    # paking vialation detection start here

    

#give the urls/paths to the cameras/videos here

if __name__ == "__main__":
    source  = '../vid/fps8/output2.avi' 
    source2 = '../vid/fps8/output3.avi'    
    source3 = '../vid/fps8/output1.avi'
    avgbackgrndpath = '../vid/fps8/avgbackground.png'    
    #source3 = '../../vd1.mp4'

    # source = 'http://root:pass@192.168.0.90/mjpg/video.mjpg' 
    # source2 = 'http://root:pass@192.168.0.100/mjpg/video.mjpg'    
    # source3 = 'http://root:pass@192.168.0.110/mjpg/video.mjpg' 
       
    run(source, source2, source3)
    

