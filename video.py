import cv2
import imutils 
import numpy as np
from PIL import Image
import keras
from keras.models import Model
from keras.layers import Dense,Conv2D,Flatten,Input,BatchNormalization,MaxPool2D,Dropout,LeakyReLU
from keras import regularizers
from keras.models import load_        model

from keras.callbacks import ReduceLROnPlateau
import pyautogui

bg=None

def resize_imaage(image_name):
    img=Image.open(image_name)
    bw=100
    wp=float(bw/img.size[1])
    h=int(wp*img.size[0])

    img = img.resize((bw,h),Image.ANTIALIAS)
    img.save(image_name)

def get_predictedclass(image_name):
    img=cv2.imread(image_name)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=img.reshape((1,96,100,1))
    imgs=[img]
    output=resnet.predict(imgs)

    return np.argmax(output),0

def show(predictedClass,confidence):
    textImage = np.zeros((300,512,3), np.uint8)
    className = ""

    if predictedClass == 0:
        className = "Fist"
        pyautogui.press("space")
    elif predictedClass == 1:
        className = "Palm"
        pyautogui.press("space")
    elif predictedClass == 2:
        className = "Thumb Left"
        pyautogui.press("left")
    elif predictedClass == 3:
        className = "Thumb Right"
        pyautogui.press("right")

    cv2.putText(textImage,"Pedicted Class : " + className, 
    (30, 30), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)

    cv2.putText(textImage,"Confidence : " + str(confidence * 100) + '%', 
    (30, 100), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)
    cv2.imshow("Statistics", textImage)


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

def main():
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
                    cv2.imwrite("Temp.png",thresholded)
                    resize_imaage("Temp.png")
                    predicted_class,confidence = get_predictedclass("Temp.png")
                    show(predicted_class,confidence)

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







inp=Input(shape=(96,100,1))
x=inp
x=BatchNormalization()(x)

x=Conv2D(filters = 4, strides = 1 ,kernel_size=(1,1),padding="same",kernel_regularizer= regularizers.l1_l2())(x)
x=Conv2D(filters = 4, strides = 1 ,kernel_size=(1,1),padding="same",kernel_regularizer= regularizers.l1_l2())(x)
x=MaxPool2D(pool_size=(1,1))(x)

x=BatchNormalization()(x)
x=Conv2D(filters = 32, strides = 1 ,kernel_size=(5,5),padding="same",kernel_regularizer= regularizers.l1_l2())(x)
x=Conv2D(filters = 32, strides = 1 ,kernel_size=(5,5),padding="same",kernel_regularizer= regularizers.l1_l2())(x)
x=MaxPool2D(pool_size=(2,2))(x)
x=BatchNormalization()(x)
x=Dropout(rate=0.2)(x)
x=LeakyReLU()(x)

x=Conv2D(filters = 64, strides = 1 ,kernel_size=(5,5),padding="same",kernel_regularizer= regularizers.l1_l2())(x)
x=Conv2D(filters = 64, strides = 1 ,kernel_size=(5,5),padding="same",kernel_regularizer= regularizers.l1_l2())(x)
x=MaxPool2D(pool_size=(2,2))(x)
x=BatchNormalization()(x)
x=Dropout(rate=0.2)(x)

x=Flatten()(x)
x=Dense(256,kernel_regularizer=regularizers.l1_l2())(x)
x=LeakyReLU()(x)
x=Dropout(rate=0.2)(x)
x=Dense(512,kernel_regularizer=regularizers.l1_l2())(x)
x=LeakyReLU()(x)
x=Dense(4,activation="softmax")(x)
resnet=Model(inp,x,name="Version1")

from keras.callbacks import ReduceLROnPlateau
rlr=ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=3,min_lr=1e-10)
callbacks=[rlr]

resnet.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
print("loading model")
resnet=load_model("model.h5")
main()