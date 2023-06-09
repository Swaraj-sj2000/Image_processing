
from sre_constants import SUCCESS
import cv2 as cv
import mediapipe as mp                           
import time

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectCon=detectCon
        self.trackCon=trackCon

        self.mpHand=mp.solutions.hands 
        self.hands=self.mpHand.Hands() 
        self.mpDraw=mp.solutions.drawing_utils  

    def findHands(self,frame,draw=True):
        imgRGB=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        
        final=self.hands.process(imgRGB)
    
        if final.multi_hand_landmarks:
            for handLm in final.multi_hand_landmarks: 
                 
                if draw:                        
                    self.mpDraw.draw_landmarks(frame,handLm,self.mpHand.HAND_CONNECTIONS)    
        return frame         

def main():
    pTime=0
    cTime=0
    cap=cv.VideoCapture(1)
    detector=handDetector()
    
    while True:
        SUCCESS,frame=cap.read()
        frame=detector.findHands(frame)
        cTime=time.time()                                                  
        fps=1/(cTime-pTime)
        pTime=cTime
        frame=cv.flip(frame,1)
        cv.putText(frame,f"{int(fps)}",(50,50),cv.FONT_HERSHEY_PLAIN,2,color=(255,0,255),thickness=2)
        cv.imshow("frame",frame)
        if cv.waitKey(1) & 0xFF==ord('x'):
            break

if __name__=="__main__":
    main()