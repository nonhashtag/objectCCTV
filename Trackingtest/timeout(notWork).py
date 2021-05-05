# -*- coding: utf-8 -*-

from imutils import build_montages
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2

#onlist -> [idx, object-N, checkedtime, startX~~~endY]
def tracker(nextidx, onlist: list, detected: list): #detected-component = [ 0, object-N, confidence, startX, startY, endX, endY ]
    if len(onlist) == 0:
        onlist.append([nextidx[0], detected[1], datetime.now(), detected[3], detected[4], detected[5], detected[6]])
        nextidx[0] += 1
    for i in range(len(onlist)):
        if (onlist[i][1] == detected[1]
                and abs(onlist[i][3]-detected[3]) < 50
                and abs(onlist[i][4]-detected[4]) < 50
                and abs(onlist[i][5]-detected[5]) < 50
                and abs(onlist[i][6]-detected[6]) < 50):
            onlist[i][2] = datetime.now()
            onlist[i][3] = detected[3]
            onlist[i][4] = detected[4]
            onlist[i][5] = detected[5]
            onlist[i][6] = detected[6]
        if i==len(onlist)-1 and not(onlist[i][1] == detected[1]
                and abs(onlist[i][3]-detected[3]) < 50
                and abs(onlist[i][4]-detected[4]) < 50
                and abs(onlist[i][5]-detected[5]) < 50
                and abs(onlist[i][6]-detected[6]) < 50):
            onlist.append([nextidx[0], detected[1], datetime.now(), detected[3], detected[4], detected[5], detected[6]])
            nextidx[0] += 1
            #onlist.append([idx, detected[1], datetime, i[3], i[4], i[5], i[6], [i[3], i[4], i[5], i[6]]])
            #ㄴ이건 사람객체에도 적용가능하게 할 예정(처음 좌표상태)



def timeout(onlist: list):
    if len(onlist) > 0:
        while True:
            
            break

        for i in range(len(onlist)):
            if (datetime.now() - onlist[i][2]).seconds > 0.5:
                del onlist[i]


imageHub = imagezmq.ImageHub()

onlist = []
nextidx = [0]

#클래스 선언
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

person_N = [0, 15, 100, 0, 0, 0, 0]
car_N = [0, 7, 100, 0, 0, 0, 0]

#네트워크 불러오기
net = cv2.dnn.readNetFromCaffe("imagezmq-streaming/MobileNetSSD_deploy.prototxt",
                               "imagezmq-streaming/MobileNetSSD_deploy.caffemodel")

#고려하는 객체 지정
CONSIDER = set(["person", "car", "bus"])
#CONSIDER = set(["car", "bus"])
objCount = {obj: 0 for obj in CONSIDER}
frameDict = {}

lastActive = {}
lastActiveCheck = datetime.now()

ESTIMATED_NUM_PIS = 4
ACTIVE_CHECK_PERIOD = 10
ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD

mW = 2
mH = 2

#video = cv2.VideoCapture("imagezmq-streaming/20210118_172040.mp4")
#video = cv2.VideoCapture("imagezmq-streaming/20210118_171949.mp4")
#video = cv2.VideoCapture("imagezmq-streaming/highway.mp4")
#video = cv2.VideoCapture("imagezmq-streaming/22.mp4")
video = cv2.VideoCapture(0)


while True:
    rpiName, frame = video.read()
    #frame = imutils.resize(frame, width=400)
    frame = imutils.resize(frame, width=480)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    timeout(onlist)
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    objCount = {obj: 0 for obj in CONSIDER}

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] in CONSIDER:
                objCount[CLASSES[idx]] += 1
                #print('시작 {} 끝'.format(CLASSES[idx])) #################################################### 테스트지점
                #print('시작 {} 끝'.format(detections[0,0,0]))
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                detected = [0, detections[0,0,i,1], datetime, startX, startY, endX, endY]
                #If it is a Person
                '''if detections[0,0,i,1]==15:
                    person_N.append()
                #If it is a Vehicle
                elif detections[0,0,i,1] <= 7:
                    car_N.append()'''
                #startX, startY, endX, endY = tracker(detections)
                tracker(nextidx, onlist, detected)
                #print('시작 {} {} {} {} 끝'.format(startX, startY, endX, endY))
                print(len(onlist))

                cv2.rectangle(frame, (startX, startY), (endX, endY),
                            (255, 0, 0), 2)

                objidx=CLASSES[idx]+" "+str(i)
                cv2.putText(frame, objidx, (startX, startY - 20), cv2.FONT_ITALIC, 0.5,
                                (255, 255, 255), 1)
                #cv2.putText(frame, rpiName, (startX, startY-20), cv2.FONT_ITALIC, 0.5,
                #            (255, 255, 255), 1)

    #Show Device ID
    #cv2.putText(frame, rpiName, (10, 25),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # draw the object count on the frame
    label = ", ".join("{}: {}".format(obj, count) for (obj, count) in
                      objCount.items())
    cv2.putText(frame, label, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    #Show Object Class Above the rectangle
    #if CLASSES[idx] in CONSIDER:
    #    cv2.putText(frame, label, (startX, startY-20), cv2.FONT_ITALIC, 0.5,
    #                (255, 255, 255), 1)

    # update the new frame in the frame dictionary
    frameDict[rpiName] = frame

    montages = build_montages(frameDict.values(), (w, h), (mW, mH))

    for (i, montage) in enumerate(montages):
        cv2.imshow("Home pet location monitor ({})".format(i),
                   frame)
                   #montage)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
