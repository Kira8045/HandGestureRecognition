import cv2
import imutils 
import numpy as np
import keras

from keras.models import Model
from keras.layers import Dense,Conv2D,Flatten,Input,BatchNormalization,MaxPool2D,Dropout,LeakyReLU
from keras import regularizers
from sklearn.utils import shuffle

bg=None


def run_avg(image,aw):
    global bg
    if bg is None:
        bg=image.copy().astype("float")
        return
    cv2.accumulateWeighted(image,bg,aw)

def segment(image,thres=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"),image)

    thresholded= cv2.threshold(diff,thres,255,cv2.THRESH_BINARY)[1]

    (cnts,_) = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    segmented = max(cnts,key = cv2.contourArea)
    return (thresholded,segmented)

if __name__=="__main__":
    aw=0.5
    camera= cv2.VideoCapture(0)

    top,right,bottom,left=50,350,300,590

    num_frames=0
    im_num=0;
    start_recording=False
    while(True):
        (grabbed,frame) = camera.read()
        frame = imutils.resize(frame,width = 700)
        frame = cv2.flip(frame , 1)
        clone = frame.copy()
        (height,width) = frame.shape[:2]

        roi = frame[top:bottom,right:left]

        gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(7,7),0)
        
        if num_frames <30:
            run_avg(gray , aw)
        else:
            hand = segment(gray)

            if hand is not None:
                (thresholded,segmented) = hand

                cv2.drawContours(clone, [segmented + (right,top)], -1, (0,0,0))
                if start_recording:
                    cv2.imwrite("Dataset/thumb_right/tr_"+str(im_num)+".png",thresholded)
                    im_num+=1
                cv2.imshow("Thresholded",thresholded)

        cv2.rectangle(clone, (left,top),(right,bottom),(0,255,0),2)
        num_frames +=1
        print(num_frames)
        cv2.imshow("Video feed",clone)

        keypress = cv2.waitKey(1) & 0xff

        if keypress == ord("q") or im_num>100:
            break
        if keypress == ord("s"):
            start_recording=True
    camera.release()
    cv2.destroyAllWindows()
