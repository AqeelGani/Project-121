import cv2
import numpy as np
import time

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))

cap = cv2.VideoCapture(0)

time.sleep(2)

bg = 0
for i in range(60):
    ret,bg = cap.read()

bg = np.flip(bg,axis = 1)

while(cap.isOpened()):
    ret,img = cap.read()
    if not ret:
        break

    img = np.flip(img,axis = 1)
    img = cv2.resize(img(640,480))
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv = cv2.resize(hsv(640,480))

    lower_red = np.array([104,153,70])
    upper_red = np.array([30,30,0])
    mask1 = cv2.inRange(hsv,lower_red,upper_red)

    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))

    mask2 = cv2.bitwise_not(mask1)

    res1 = cv2.bitwise_and(img,img,mask = mask2)
    res2 = cv2.bitwise_and(bg,bg,mask = mask1)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0.0)

    output_file.write(final_output)
    cv2.imshow('cloak',final_output)

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()