import cv2
import numpy as np
import imutils

def detect_plate(cap):
    cropped=None
    CONFIDENCE = 0.9
    THRESHOLD = 0.3
    LABELS = ['Car', 'Plate']

    #cap = cv2.imread('imgs/06.PNG')
    cap = imutils.resize(cap, width = 800)
    #cap = cv2.resize(cap, None, fx=4, fy=4)
    net = cv2.dnn.readNetFromDarknet('cfg/yolov4-ANPR.cfg', 'yolov4-ANPR.weights')


    colors = np.random.uniform(0, 255, size=(len(LABELS), 3))

    height, width, channels = cap.shape

    blob = cv2.dnn.blobFromImage(cap, scalefactor=1/255., size=(416, 416), swapRB=True)
    net.setInput(blob)
    output = net.forward()

    boxes, confidences, class_ids = [], [], []

    for det in output:
        box = det[:4]
        scores = det[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > CONFIDENCE:
            cx, cy, w, h = box * np.array([width, height, width, height])
            x = cx - (w / 2)
            y = cy - (h / 2)

            boxes.append([int(x), int(y), int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)


    there_plate = 0
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(cap, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
            cv2.putText(cap, text='%s %.2f %d' % (LABELS[class_ids[i]], confidences[i], w), org=(x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
            #print('%s %.2f %d' % (LABELS[class_ids[i]], confidences[i], w), end=" ")
            #print(boxes[i])
            if(LABELS[class_ids[i]])== "Plate":
                cropped = cap[y:y+h, x:x+w]
                return cropped
    return False