#실시간 영상저장테스트
import datetime
import cv2
import imutils
import time

capture = cv2.VideoCapture("_DSC3358.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
record = False
start_time = time.time()

while True:
    if (capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
        capture.open("_DSC3358.mp4")

    ret, frame = capture.read()
    frame = imutils.resize(frame, width=400)
    cv2.imshow("VideoFrame", frame)

    if record == False:
        print("녹화 시작")
        record = True
        video = cv2.VideoWriter(time.strftime('%Y-%m-%d %H시 %M분', time.localtime(time.time())) + '.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    if ((time.time() - start_time) % 5 >= 0 and (time.time() - start_time) % 5 < 0.2):
        print("녹화 중지")
        record = False
        video.release()
    if record == True:
        print("녹화 중..")
        video.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
