# -*- coding: utf-8 -*-

from plating import detect_plate as deplate
from imutils import build_montages
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2
import time

from twilio.rest import Client


'''def sns(onlist: list, sent: list):
    if len(sent)==0:
        return
    if any(i in onlist for i in sent):
        a = sent.index(i)'''




#onlist -> [idx, object-N, firsttime, uptime, startX~~~~endY]l
def tracker(nextidx, onlist: list, detected: list): #detected-component = [ 0, object-N, confidence, startX, startY, endX, endY ]
    if len(onlist) == 0:
        onlist.append([nextidx[0], detected[1], datetime.now(), datetime.now(), detected[3], detected[4], detected[5], detected[6],
                       False])
        nextidx[0] += 1
    for i in range(len(onlist)):
        if (onlist[i][1] == detected[1]
                and abs(onlist[i][4]-detected[3]) < 80
                and abs(onlist[i][5]-detected[4]) < 80
                and abs(onlist[i][6]-detected[5]) < 80
                and abs(onlist[i][7]-detected[6]) < 80):
            onlist[i][3] = datetime.now()
            onlist[i][4] = detected[3]
            onlist[i][5] = detected[4]
            onlist[i][6] = detected[5]
            onlist[i][7] = detected[6]
            break
        if i==len(onlist)-1 and not(onlist[i][1] == detected[1]
                and abs(onlist[i][4]-detected[3]) < 80
                and abs(onlist[i][5]-detected[4]) < 80
                and abs(onlist[i][6]-detected[5]) < 80
                and abs(onlist[i][7]-detected[6]) < 80):
            onlist.append([nextidx[0], detected[1], datetime.now(), datetime.now(), detected[3], detected[4], detected[5], detected[6],
                           False])
            nextidx[0] += 1
            #onlist.append([idx, detected[1], datetime, i[3], i[4], i[5], i[6], [i[3], i[4], i[5], i[6]]])
            #ㄴ이건 사람객체에도 적용가능하게 할 예정(처음 좌표상태)

def timeout(onlist: list):
    to_del = []
    if len(onlist) > 0:
        for i in range(len(onlist)):
            if (datetime.now() - onlist[i][3]).seconds > 0.5:
                to_del.append(i)
                #print(to_del)
    to_del.reverse()
    if len(to_del) > 0:
        for i in range(len(to_del)):
            #print('시작 {} 끝'.format(to_del[i]))
            del onlist[to_del[i]]

def indexing(onlist :list, detected: list) -> str:
    for i in onlist:
        if i[1]==detected[1] and i[4:8]==detected[3:]:
            return str(i[0])
    return "???"

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

account_sid = 'ACf0a587ebd35478480cd8fc445062e935'
auth_token = 'c7ee620178aeee3d6a45d8f7ae881a00'
client = Client(account_sid, auth_token)


#video = cv2.VideoCapture("imagezmq-streaming/20210118_172040.mp4")
#video = cv2.VideoCapture("imagezmq-streaming/20210118_171949.mp4")
#video = cv2.VideoCapture("imagezmq-streaming/highway.mp4")
video = cv2.VideoCapture("cams/20210510_145505.mp4")
#video = cv2.VideoCapture("cams/20210510_143427.mp4") #2번 주정차 1+1
#video = cv2.VideoCapture(0)



start_time = time.time()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
record = False
while True:
    rpiName, frame = video.read()
    #frame = imutils.resize(frame, width=400)
    frame = imutils.resize(frame, width=1280)
    (h, w) = frame.shape[:2]


    """#녹화
    if record == False:
        # print("녹화 시작")
        record = True
        capture = cv2.VideoWriter('save_video/'+time.strftime('%Y-%m-%d %H시 %M분', time.localtime(time.time())) + '.avi', fourcc, 10.0,
                                  (frame.shape[1], frame.shape[0]))
    if ((time.time() - start_time) % 60 >= 0 and (time.time() - start_time) % 60 < 0.2):
        # print("녹화 중지")
        record = False
        capture.release()
    if record == True:
        # print("녹화 중..")
        capture.write(frame)"""


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
        #print("{}타입니다!!!".format(detections))

        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] in CONSIDER:
                objCount[CLASSES[idx]] += 1
                #print('시작 {} 끝'.format(CLASSES[idx])) #################################################### 테스트지점
                #print('시작 {} 끝'.format(detections[0,0,0]))
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                detected = [0, detections[0,0,i,1], datetime, startX, startY, endX, endY]

                #startX, startY, endX, endY = tracker(detections)
                tracker(nextidx, onlist, detected)
                #print('시작 {} {} {} {} 끝'.format(startX, startY, endX, endY))
                #print(len(onlist))
                #print(objCount[CLASSES[idx]])

                cv2.rectangle(frame, (startX, startY), (endX, endY),
                            (255, 0, 0), 2)

                objidx=CLASSES[idx] + " " + indexing(onlist, detected)
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

    #현황판
    #cv2.putText(frame, label, (10, h - 20),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


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
    cropped = False
    if len(onlist) > 0:
        for i in range(len(onlist)):
            if (datetime.now() - onlist[i][2]).seconds >= 10 and onlist[i][1] == 7:
                if onlist[i][8] == False:
                    try:
                        print("문자")
                        print("차량 경고 {}번 차량".format(onlist[i][0]))
                        cropped = frame[onlist[i][5]:onlist[i][7], onlist[i][4]:onlist[i][6]] #주차차량만 잘라내기
                        pnow = time.strftime('%Y-%m-%d %H_%M', time.localtime(time.time()))
                        cv2.imwrite('parking_shot/'+pnow+'.jpg', cropped) #주차차량 이미지 저장
                        #find_plate = deplate(cropped)
                        #if find_plate != None:
                        #    cv2.imwrite('car_plates/'+pnow+'.jpg', find_plate)
                        #client.api.account.messages.create(to="+821022302480", from_="+12513049647", body="Car {} Warn!".format(onlist[i][0]))
                        #client.api.account.messages.create(to="+821022302480", from_="+12513049647", body="Car {} Warn!".format(i[0]))
                        onlist[i][8] = True
                    except AttributeError as e:
                        print(e)

    if not(cropped is False):
        cv2.imshow("testing", cropped)


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
	
