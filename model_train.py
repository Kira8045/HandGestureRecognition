import keras
import cv2
from PIL import Image
from keras.models import Model
from keras.layers import Dense,Conv2D,Flatten,Input,BatchNormalization,MaxPool2D,Dropout,LeakyReLU
from keras import regularizers
from sklearn.utils import shuffle
import numpy as np
imgs=[]
for i in range(100):
    image=cv2.imread(r"Dataset\fist\fist_"+str(i)+".png")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    imgs.append(image.reshape(96,100,1))
for i in range(100):
    image=cv2.imread(r"Dataset\palm\palm_"+str(i)+".png")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    imgs.append(image.reshape(96,100,1))
for i in range(100):
    image=cv2.imread(r"Dataset\thumb_left\tl_"+str(i)+".png")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    imgs.append(image.reshape(96,100,1))
for i in range(100):
    image=cv2.imread(r"Dataset\thumb_right\tr_"+str(i)+".png")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    imgs.append(image.reshape(96,100,1))

output=[]
for i in range(100):
    output.append([1,0,0,0])
    
for i in range(100):
    output.append([0,1,0,0])
for i in range(100):
    output.append([0,0,1,0])
for i in range(100):
    output.append([0,0,0,1])

imgs=np.array(imgs)
output=np.array(output)
imgs,output=shuffle(imgs,output,random_state=0)

from sklearn.model_selection import train_test_split as spt
train_x,val_x,train_y,val_y=spt(imgs,output,test_size=0.01)

from keras.preprocessing.image import ImageDataGenerator as IMGG
datagen=IMGG(rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    rescale=0.3,
    horizontal_flip=False,
    vertical_flip=False)
datagen.fit(train_x)



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

resnet.fit_generator(datagen.flow(train_x,train_y,batch_size=64),epochs=30,validation_data=(val_x,val_y),steps_per_epoch=256,callbacks=callbacks)

resnet.save("model.h5")

