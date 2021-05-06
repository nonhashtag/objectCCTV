from imutils import build_montages
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2
import time
from twilio.rest import Client


#onlist -> [idx, object-N, checkedtime, startX~~~~endY]
sms = False

def tracker(nextidx, onlist: list, detected: list): #detected-component = [ 0, object-N, confidence, startX, startY, endX, endY ]
    if len(onlist) == 0:
        onlist.append([nextidx[0], detected[1], datetime.now(), datetime.now(), detected[3], detected[4], detected[5], detected[6]])
        nextidx[0] += 1
    for i in range(len(onlist)):
        if (onlist[i][1] == detected[1]
                and abs(onlist[i][4]-detected[3]) < 30
                and abs(onlist[i][5]-detected[4]) < 30
                and abs(onlist[i][6]-detected[5]) < 30
                and abs(onlist[i][7]-detected[6]) < 30):
            onlist[i][3] = datetime.now()
            onlist[i][4] = detected[3]
            onlist[i][5] = detected[4]
            onlist[i][6] = detected[5]
            onlist[i][7] = detected[6]
        if i==len(onlist)-1 and not(onlist[i][1] == detected[1]
                and abs(onlist[i][4]-detected[3]) < 30
                and abs(onlist[i][5]-detected[4]) < 30
                and abs(onlist[i][6]-detected[5]) < 30
                and abs(onlist[i][7]-detected[6]) < 30):
            onlist.append([nextidx[0], detected[1], datetime.now(), datetime.now(), detected[3], detected[4], detected[5], detected[6]])
            nextidx[0] += 1
            #onlist.append([idx, detected[1], datetime, i[3], i[4], i[5], i[6], [i[3], i[4], i[5], i[6]]])
            #ㄴ이건 사람객체에도 적용가능하게 할 예정(처음 좌표상태)


def timeout(onlist: list):
    to_del = []
    if len(onlist) > 0:
        for i in range(len(onlist)):
            if (datetime.now() - onlist[i][3]).seconds > 1:
                to_del.append(i)
                print(to_del)
    to_del.reverse()
    if len(to_del) > 0:
        for i in range(len(to_del)):
            print('시작 {} 끝'.format(to_del[i]))
            del onlist[to_del[i]]

def indexing(onlist :list, detected: list) -> int:
    for i in onlist:
        if i[1]==detected[1] and i[4:]==detected[3:]:
            return str(i[0])

imageHub = imagezmq.ImageHub()

onlist = []
nextidx = [0]
#클래스 선언
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
"sofa", "train", "tvmonitor"]

#네트워크 불러오기
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

#고려하는 객체 지정
CONSIDER = set(["person", "car", "bus"])
objCount = {obj: 0 for obj in CONSIDER}
frameDict = {}

lastActive = {}
lastActiveCheck = datetime.now()
p_time = datetime.now()
c_time = datetime.now()

ESTIMATED_NUM_PIS = 4
ACTIVE_CHECK_PERIOD = 10
ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD

mW = 2
mH = 2

account_sid = 'AC2c1c70d434d0540600123fe01d1847cb'
auth_token = 'e54cb5c44f63510684c0b54dc06439cc'
client = Client(account_sid, auth_token)


record = False
start_time = time.time()
fourcc = cv2.VideoWriter_fourcc(*'XVID')

#video = cv2.VideoCapture("rearview_driving_mounts_1080p.mp4")
video = cv2.VideoCapture("Stop_sign.mp4")
#video = cv2.VideoCapture("FroggerHighway.mp4")
#video = cv2.VideoCapture("testvideo.mp4")

while True:
    rpiName, frame = video.read()
    ret, frame1 = video.read()
    if record == False:
        #print("녹화 시작")
        record = True
        capture = cv2.VideoWriter(time.strftime('%Y-%m-%d %H시 %M분', time.localtime(time.time())) + '.avi', fourcc, 10.0,
                                (frame1.shape[1], frame1.shape[0]))
    if ((time.time() - start_time) % 60 >= 0 and (time.time() - start_time) % 60 < 0.2):
        #print("녹화 중지")
        record = False
        capture.release()
    if record == True:
        #print("녹화 중..")
        capture.write(frame1)

    frame = imutils.resize(frame, width=400)
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

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                detected = [0, detections[0, 0, i, 1], datetime, startX, startY, endX, endY]
                # If it is a Person
                '''if detections[0,0,i,1]==15:
                    person_N.append()
                #If it is a Vehicle
                elif detections[0,0,i,1] <= 7:
                    car_N.append()'''
                # startX, startY, endX, endY = tracker(detections)
                tracker(nextidx, onlist, detected)
                # print('시작 {} {} {} {} 끝'.format(startX, startY, endX, endY))
                print(len(onlist))

                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (255, 0, 0), 2)

                objidx = CLASSES[idx] + " " + indexing(onlist, detected)
                cv2.putText(frame, objidx, (startX, startY - 20), cv2.FONT_ITALIC, 0.5,
                            (255, 255, 255), 1)


#cv2.putText(frame, rpiName, (startX, startY-20), cv2.FONT_ITALIC, 0.5,
# (255, 255, 255), 1)

#cv2.putText(frame, rpiName, (10, 25),
# cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# draw the object count on the frame
    label = ", ".join("{}: {}".format(obj, count) for (obj, count) in objCount.items())
    cv2.putText(frame, label, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#Show Object Class Above the rectangle
    #cv2.putText(frame, label, (startX, startY-20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)

# update the new frame in the frame dictionary
    frameDict[rpiName] = frame

    montages = build_montages(frameDict.values(), (w, h), (mW, mH))

    for (i, montage) in enumerate(montages):
        cv2.imshow("Monitor ({})".format(i), montage)
        key = cv2.waitKey(1) & 0xFF

    '''if objCount['person'] != 0:
        person_t = (datetime.now() - p_time)
        print("person_t : {}".format(person_t))
        if person_t.seconds >= 10.0:
            print("사람 경고")
            if sms == False :
                client.api.account.messages.create(to="+821095962543", from_="+17326064954", body="Person Warn!")
                sms = True
    else:
        p_time = datetime.now()
        sms = False'''
    if len(onlist) > 0:
        for i in onlist:
            if (datetime.now() - i[2]).seconds >= 2 and i[1] == 7:
                print("차량 경고 {}번 차량".format(i[0]))
                #if i[8] == False:
                #    print("문자")
                #    client.api.account.messages.create(to="+821095962543", from_="+17326064954", body="Car {} Warn!".format(i[0]))
                #    i[8] = True
            else:
                #c_time = datetime.now()
                sms = False
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
