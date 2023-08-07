import cv2
import numpy as np

thres = 0.50  # Threshold to detect object
nms_threshold = 0.2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 150)

# Exporting the output as video
ret, img = cap.read()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('output.avi', fourcc, 24, (img.shape[1], img.shape[0]))

classNames = []
classFile = "coco.names"

with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)    

font = cv2.FONT_HERSHEY_COMPLEX

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    if len(indices)>0:
        for classId, confidence, box in zip(classIds.flatten(),confs,bbox):

            for i in indices:
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
                cv2.putText(img,classNames[classIds[i]-1],(box[0] + 10, box[1] + 30),font, fontScale = 1, color=(0, 255, 0), thickness=2)
                cv2.putText(img,str(round (confidence*100, 1)),(box[0] +160, box[1]+30),font, fontScale = 1, color=(0, 255, 0), thickness=2)

    video.write(img)
    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and destroy the window
cap.release()
cv2.destroyAllWindows()
