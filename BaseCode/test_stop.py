import cv2
import numpy as np

cap = cv2.VideoCapture("source_from_nto/BaseCode/area.mp4")
minb, ming, minr, maxb, maxg, maxr = 22, 67, 96, 255, 255, 255
AREA  = False
once = True

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,(minb,ming,minr),(maxb,maxg,maxr))

    hist = np.sum(mask, axis=1)
    maxStrInd = np.argmax(hist)
    ind = hist[maxStrInd]//255

    if ind > 300 and once:
        AREA = True
        once = False

        
    else:  AREA = False
    print(AREA)
    #hist = np.sum(mask)
    # print(hist)
    # if hist > 28303725:
    #     AREA = True

    result = cv2.bitwise_and(frame,frame,mask=mask)

    # maxStrInd = np.argmax(hist)
    # print(maxStrInd)


   

    cv2.imshow('result1', result)
    #cv2.imshow('frame_input', frame)
    #cv2.imshow('mask', mask)
    k = cv2.waitKey(100)
    if k == ord('q'):
        break
   
    
  