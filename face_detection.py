import cv2 as cv
import numpy
# img=cv.imread("2_faces.jpg")
# gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# detector=cv.CascadeClassifier("haar_face.xml")
# faces_rect=detector.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=3)
# print(f"{len(faces_rect)}")
# for (x,y,w,h) in faces_rect:
#     cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
# cv.imshow('detected',img)
# cv.waitKey(0)

vid=cv.VideoCapture(1)
while True:
    isTrue,frame=vid.read()
    frame=cv.flip(frame,1)
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    #facedetection
    detector=cv.CascadeClassifier("haar_face.xml")

    faces_rect=detector.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=3)

    for (x,y,w,h) in faces_rect:
        # print(x,y,w,h,type(x),type(y),type(w),type(h))
        if(len(faces_rect)>1):
            r=(0,0,255)
        else:r=(0,255,0)
        cen=(numpy.intc(int(x)+int(w)/2),numpy.intc(int(y)+int(h)/2))
        rad=numpy.intc(int(h)/2+int(w)/2-47)
        cv.circle(frame,cen,rad,r,thickness=1)
        # cv.rectangle(frame,(x,y),(x+w,y+h),r,thickness=2 )
        tx=int(x)+int(w)/2-20
        ty=int(y)-20
        cv.putText(frame,"Face detected ",(numpy.intc(tx),numpy.intc(ty)),cv.FONT_HERSHEY_TRIPLEX,0.3,(255,0,0),thickness=1)
        print(x,y,w,h)
    #eyedetection
    eyesdetector=cv.CascadeClassifier("eye_cascade.xml").detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
    for(ex,ey,ew,eh) in eyesdetector:
        cv.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(255,255,255),thickness=1)

    cv.imshow("detected",frame)
    
    if cv.waitKey(1) & 0xFF==ord('x'):
        break
vid.release()
cv.destroyAllWindows()