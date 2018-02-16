import numpy as np
import cv2
import sys
import caffe
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import math
import pylab

import time
					
import collections
import itertools
import tensorflow as tf
import functions
import deep_net

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



from sklearn.preprocessing import normalize
caffe.set_mode_gpu()
cap=cv2.VideoCapture('../../vd1.mp4')
net = caffe.Net('../../enet_deploy_final.prototxt','../../segnet_iter_55000.caffemodel',caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

Sky = [255,0,0]
Building = [0,255,0]
Pole = [0,0,255]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]
label_colours = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
label_colour = np.array([2, 1, 0])
cropped=[]
resizeimg=[]
while(True):
    #ret=cap.set(cv2.CAP_PROP_FPS,4)
    ret,frame=cap.read()
    #ret=cap.set(3,360)
    #ret=cap.set(4,480)
    #frame = cv2.imread("myimage.png",1)
    s=cv2.resize(frame,(480,360))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0)) # if using RGB instead of BGR
    transformer.set_raw_scale('data', 255.0)
    net.blobs['data'].reshape(1,3,360,480)
    net.blobs['data'].data[...] = transformer.preprocess('data', s)
    output = net.forward()
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
    NoAllVContours = len(hierarchy[0])
    for i in range(NoAllVContours):    
        M = cv2.moments(contours[i])
        if (M['m00']>2500):
            x,y,w,h=cv2.boundingRect(contours[i])
            cropped = s[y:y+h, x:x+w]
            resizeimg=cv2.resize(cropped,(256,32))
            cv2.imshow('framerty',resizeimg)
            im_gray = cv2.cvtColor(resizeimg, cv2.COLOR_BGR2GRAY) / 255.
            f = np.load("weights.npz")
            param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]
            letter_probs = detect(im_gray, param_vals)
            code = letter_probs_to_code(letter_probs)
            print("Number Plate ->",code)


    #print cap.get(cv2.CAP_PROP_FPS)
    #print frame.shape
    cv2.imshow('frame',s)
    cv2.imshow('frame2',rgb)
    if cv2.waitKey(1) & 0xff== ord('q'):
        break
cap.release()
cv2.destroyAllwindows()

