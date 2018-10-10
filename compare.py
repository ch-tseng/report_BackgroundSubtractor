#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import imutils
import time

bgPath = "output/1538990680298.jpg"
cap = cv2.VideoCapture('/media/sf_VMshare/videos/t1.mp4')
#cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
print(width, height)

displayResize = 600
FILE_OUTPUT = '/media/sf_VMshare/all.avi'
minArea = 6000

fgbg_mog = cv2.bgsegm.createBackgroundSubtractorMOG()
#fgbg_mog = cv2.createBackgroundSubtractorMOG(history=60, detectShadows=True)
fgbg_mog2 = cv2.createBackgroundSubtractorMOG2( detectShadows=True)
fgbg_knn = cv2.createBackgroundSubtractorKNN( detectShadows=True)
fgbg_gmg = cv2.bgsegm.createBackgroundSubtractorGMG()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(FILE_OUTPUT,fourcc, 30.0, (int(width)*2,int(height*3)))


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.GaussianBlur(img, (47, 47), 0)

    return img

def posprocess(img):
    (T, img) = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
    #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #    cv2.THRESH_BINARY,11,2)
    img = cv2.dilate(img, None, iterations=42)
    img = cv2.erode(img, None, iterations=42)
    img = cv2.dilate(img, None, iterations=8)

    return img

def findContours(img):
    _, cnts, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return cnts

lastFrame = None
while True:
    ret, frame = cap.read()
    frame_preprocess = preprocess(frame)

    if(lastFrame is None): lastFrame = frame_preprocess
    fgmask_diff = cv2.absdiff(frame_preprocess, lastFrame)
    lastFrame = frame_preprocess

    fgmask_mog = fgbg_mog.apply(frame_preprocess)
    fgmask_mog2 = fgbg_mog2.apply(frame_preprocess)
    fgmask_knn = fgbg_knn.apply(frame_preprocess)
    fgmask_gmg = fgbg_gmg.apply(frame_preprocess)

    fgmask_mog = cv2.morphologyEx(fgmask_mog, cv2.MORPH_OPEN, kernel)
    fgmask_mog2 = cv2.morphologyEx(fgmask_mog2, cv2.MORPH_OPEN, kernel)
    fgmask_knn = cv2.morphologyEx(fgmask_knn, cv2.MORPH_OPEN, kernel)
    fgmask_gmg = cv2.morphologyEx(fgmask_gmg, cv2.MORPH_OPEN, kernel)

    boundcolor = (255,255,255)
    fontsize = 3.5
    locY = 120
    b = np.zeros(fgmask_mog.shape[:2], dtype = "uint8")
    r = np.zeros(fgmask_mog.shape[:2], dtype = "uint8")

    fgmask_diff_rgb = cv2.merge([b, fgmask_diff, r])
    cv2.putText(fgmask_diff_rgb, "cv2.absdiff", (760, locY), cv2.FONT_HERSHEY_COMPLEX, fontsize, boundcolor, 2)

    fgmask_mog_rgb = cv2.merge([b, fgmask_mog, r])
    cv2.putText(fgmask_mog_rgb, "MOG", (860, locY), cv2.FONT_HERSHEY_COMPLEX, fontsize, boundcolor, 2)

    fgmask_mog2_rgb = cv2.merge([b, fgmask_mog2, r])
    cv2.putText(fgmask_mog2_rgb, "MOG2", (860, locY), cv2.FONT_HERSHEY_COMPLEX, fontsize, boundcolor, 2)

    fgmask_knn_rgb = cv2.merge([b, fgmask_knn, r])
    cv2.putText(fgmask_knn_rgb, "KNN", (860, locY), cv2.FONT_HERSHEY_COMPLEX, fontsize, boundcolor, 2)

    fgmask_gmg_rgb = cv2.merge([b, fgmask_gmg, r])
    cv2.putText(fgmask_gmg_rgb, "GMG", (860, locY), cv2.FONT_HERSHEY_COMPLEX, fontsize, boundcolor, 2)

    cv2.putText(frame, "Original", (840, locY), cv2.FONT_HERSHEY_COMPLEX, fontsize, boundcolor, 2)

    '''
    binImg = posprocess(fgmask)
    #cv2.imshow("fgbg_posprocess", imutils.resize(fgmask, width=displayResize))
    b = np.zeros(binImg.shape[:2], dtype = "uint8")
    r = np.zeros(binImg.shape[:2], dtype = "uint8")

    binImg_rgb = cv2.merge([b, binImg, r])
    #cv2.imshow("3 Channels", imutils.resize(merged, width=displayResize))

    cnts = findContours(binImg)

    QttyOfContours = 0
    for c in cnts:
        #if a contour has small area, it'll be ignored
        if cv2.contourArea(c) < minArea:
            continue

        QttyOfContours = QttyOfContours+1    

        #draw an rectangle "around" the object
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
        cv2.rectangle(fgmask_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(binImg_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)

        #find object's centroid
        #CoordXCentroid = int((x+x+w)/2)
        #CoordYCentroid = int((y+y+h)/2)
        #ObjectCentroid = (CoordXCentroid,CoordYCentroid)
        #cv2.circle(frameOrg, ObjectCentroid, 1, (0, 0, 0), 5)
        #cv2.circle(fgmask, ObjectCentroid, 1, (0, 0, 0), 5)
    '''

    print(frame.shape, fgmask_diff_rgb.shape)
    combined1 = np.hstack((frame, fgmask_mog_rgb))
    combined2 = np.hstack((fgmask_mog2_rgb, fgmask_diff_rgb))
    combined3 = np.hstack((fgmask_knn_rgb, fgmask_gmg_rgb))
    combined = np.vstack((combined1, combined2, combined3))
    print(combined.shape)
    #combined = imutils.resize(combined, width=1980)
    cv2.imshow("Combined", imutils.resize(combined, width=displayResize))
    out.write(combined)


    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()

