import numpy as np
import cv2
import sys
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print (sys.argv[1])
cv2.imwrite("./faces/"+sys.argv[1]+".jpg", frame)
cap.release()
