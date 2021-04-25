from imutils import build_montages
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2
import time
from twilio.rest import Client

imageHub = imagezmq.ImageHub()

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
sms = False

record = False
start_time = time.time()
fourcc = cv2.VideoWriter_fourcc(*'XVID')

video = cv2.VideoCapture(0)


while True:
    rpiName, frame = video.read()
    ret, frame1 = video.read()
    if record == False:
        print("녹화 시작")
        record = True
        capture = cv2.VideoWriter(time.strftime('%Y-%m-%d %H시 %M분', time.localtime(time.time())) + '.avi', fourcc, 10.0,
                                (frame1.shape[1], frame1.shape[0]))
    if ((time.time() - start_time) % 60 >= 0 and (time.time() - start_time) % 60 < 0.2):
        print("녹화 중지")
        record = False
        capture.release()
    if record == True:
        print("녹화 중..")
        capture.write(frame1)

    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
    0.007843, (300, 300), 127.5)

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

                cv2.rectangle(frame, (startX, startY), (endX, endY),
                (255, 0, 0), 2)
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
        cv2.imshow("Home pet location monitor ({})".format(i), montage)
        key = cv2.waitKey(1) & 0xFF

    if objCount['person'] != 0:
        person_t = (datetime.now() - p_time)
        print("person_t : {}".format(person_t))
        if person_t.seconds >= 10.0:
            print("사람 경고")
            '''if sms == False :
                client.api.account.messages.create(to="+821095962543", from_="+17326064954", body="Person Warn!")
                sms = True'''
    else:
        p_time = datetime.now()
        sms = False

    if objCount['car'] != 0:
        car_t = (datetime.now() - c_time)
        print("car_t : {}".format(car_t))
        if car_t.seconds >= 10.0:
            print("차량 경고")
            '''if sms == False:
                client.api.account.messages.create(to="+821095962543", from_="+17326064954", body="Car Warn!")
                sms = True'''
    else:
        c_time = datetime.now()
        sms = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()