import math
from tkinter import font
import cv2 as cv
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np

cap = cv.VideoCapture('Videos/vid (6).mp4')

myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 0, 'smin': 152, 'vmin': 0, 'hmax': 19, 'smax': 255, 'vmax': 255}

#varibales
posListX,posListY = [],[]
xList = [item for item in range(0,1300)]
predict = False

while True:
    ss,img = cap.read()
    # img = cv.imread("Ball.png")

    img = img[0:900,:]
   
    
    #find the ball color
    imgColor, mask = myColorFinder.update(img,hsvVals)

    imgContours,contours = cvzone.findContours(img,mask,minArea=500)

    if contours:
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])
    
    if posListX:
        A, B, C = np.polyfit(posListX,posListY,2)

        for i,(posX,posY) in enumerate(zip(posListX,posListY)):
            pos = (posX,posY)
            cv.circle(imgContours,pos,8,(0,0,255),-1)
            if i==0:
                cv.line(imgContours,pos,pos,(0,255,0),4)
            else: 
                cv.line(imgContours,pos,(posListX[i-1],posListY[i-1]),(0,255,0),2)

        for x in xList:
            y = int(A * x ** 2+ B * x   + C)
            cv.circle(imgContours,(x,y),2,(255,0,255),-1)

    #prediction
    # X value 330 to 430 Y value 590
        if len(posListX)<10:
            a = A
            b = B
            c = C - 590

            x = int((-b-math.sqrt(b**2-(4*a*c)))/(2*a))
            predict = 325<x<430        
        if predict :
            cvzone.putTextRect(imgContours,"Basket",(50,150),scale=5,thickness=5,colorR=(0,250,0),offset=20)
        else:
            cvzone.putTextRect(imgContours,"No Basket",(50,150),scale=5,thickness=5,colorR=(0,0,250),offset=20)

    #display image
    imgContours = cv.resize(imgContours,(0,0),None,0.7,0.7)
    cv.imshow("ImageColor",imgContours)
    cv.waitKey(110)