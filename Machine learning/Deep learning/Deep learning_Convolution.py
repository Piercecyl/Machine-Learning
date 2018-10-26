import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle

pickle_in = open("X.pickle","rb") #打開X(圖片內容)
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")#打開Y(類別)
y = pickle.load(pickle_in)


#正規化(裡面包含0~255位)
X = X/255.0

#CNN建立
model = Sequential()
#Layer 1
model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Layer 2
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Layer 3
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))

#Layer 4
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#validation_split：案一定的比例從training 中取出來驗證，70/30
model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3)