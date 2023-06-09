import cv2 
import cv2.aruco as aruco
import numpy as np
# import os

def rescaleFrame(frame,scale=0.75):                                          #resized the photo or video using a user defined fn.
    width=int(frame.shape[1]*scale)                                          #resized times the scale
    height=int(frame.shape[0]*scale)                                                       
    dimensions=(width,height)
    return cv2.resize(frame,dimensions,interpolation=cv2.INTER_AREA)          #resize fn is built in cv


def findArucomarkers(img,markerSize=5,totalMarkers=250,draw=True):
    imggrey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key=getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucodict=aruco.Dictionary_get(key)
    arucoParam=aruco.DetectorParameters_create()
    bboxes,ids,rejected=aruco.detectMarkers(imggrey,arucodict,parameters=arucoParam)
    # print(ids,end=" ")
    if draw:
        aruco.drawDetectedMarkers(img,bboxes)
    return [bboxes,ids]

def point(image,x,y,bl,gr,re):
    cv2.circle(image,(int(x),int(y)),5,(bl,gr,re),thickness=-1)


def main():
    # cap=cv2.VideoCapture(0)
    # while True:
    #     isTrue,img=cap.read()
    #     foundAruco=findArucomarkers(img)
    #     cv2.imshow("image",img)
           
    cap=rescaleFrame(cv2.imread("test_image1.png"),scale=2)
    # img="marker.png"
    # cap=cv2.imread(img)
    foundAruco=findArucomarkers(cap)
    # os.chdir(r'D:\python projects')
    # cv2.imwrite("markerdetected.jpg", img)
    print(foundAruco[0])                                                                                                                                                                                       
    cv2.imshow("image",cap)
    cv2.waitKey(0)
    
        # if cv2.waitKey(1) and 0XFF==ord('x'):
        #    break

if __name__=="__main__":
    main()