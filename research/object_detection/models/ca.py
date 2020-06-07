import numpy as np
import cv2
cap = cv2.VideoCapture(0)
while(True):
    ret, image = cap.read()
    img_color = cv2.cvtColor(image, 0)
    cv2.imshow('window', img_color)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
