import numpy as np
import time
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 1080)

while True:

    ret, frame = cap.read()

    h, w = frame.shape[:2]

    

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
 
    if key == ord("q"):
        break

cv2.destroyAllWindows()