import numpy as np
from ultralytics import YOLO
import cv2 as cv
from utils import CvFpsCalc

CONF = 0.25 # Minimum confidence that create a bounding boxe
IOU = 0.7   # Intersection over union

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
print(type(model))

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cvFpsCalc = CvFpsCalc(buffer_len=20)  # buffer for the fps management

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    fps = cvFpsCalc.get() #counting fps
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Added the fps count in the showed image
    cv.putText(frame, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv.LINE_AA)

    # run inference on the current frame
    results = model(source=frame, show=False, save=False, conf=CONF, iou=IOU)
    #print(type(results[0]))
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        cv.imshow('frame', im_array)



    # Display the resulting frame
    #cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()