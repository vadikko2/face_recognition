import numpy as np
import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imwrite("cap.png", frame)
cap.release()
