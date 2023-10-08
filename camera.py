#source: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
#https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
#https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
#https://stackoverflow.com/questions/44588279/find-and-draw-the-largest-contour-in-opencv-on-a-specific-color-python

'''
Improvements:
add bounding box of largest contour
add gaussian blur
change s and v min to 30
'''
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

while(True):
    # Take each frame
    _, frame = cap.read()
    blur = cv.GaussianBlur(frame,(5,5),0)
    # Convert BGR to HSV
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    ret,thresh = cv.threshold(mask,127,255,0)
    contours,hierarchy = cv.findContours(thresh, 1, 2)
    cnt = max(contours, key = cv.contourArea)
    area = cv.contourArea(cnt)
    print(area)
    x,y,w,h = cv.boundingRect(cnt)
    cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()