import numpy as np
import time
import cv2
from tensorflow.keras.models import model_from_json
from PIL import Image
import sklearn


with open("model.json", "r") as fp:
    model_test = model_from_json(fp.read())

model_test.load_weights("best.h5")

net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

cap = cv2.VideoCapture(0) 
cap.set(3, 720)
cap.set(4, 1280)

while True:
    ret, frame = cap.read()

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    crop_img = cv2.imread("download.png")

    for i in range(0, detections.shape[2]):
        try:

            confidence = detections[0, 0, i, 2]

            if confidence < 0.2:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            
            crop_img = frame[startY:endY, startX:endX]


            img_array = crop_img

            img_array = cv2.resize(img_array, dsize=(140,160), interpolation=cv2.INTER_LINEAR)

            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

            result = model_test.predict(img_array.reshape(1,160,140,1))

            # print(result)

            blue_color = (255,0,0)
            red_color = (0,0,255)

            if result[0][0] > 0.90:
                color = blue_color
            else:
                color = red_color

            text = "{:.2f}%".format(result[0][0] * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
        except:
            pass

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    if key == ord("q"):
        break

cv2.destroyAllWindows()
