#%%
import os 
import numpy as np
import cv2

classes = os.listdir("train")
numberOfClasses = len(classes)

path = r"C:/Users/mrtkr/Desktop/Happiness Detection/train"

images = []
outputs = []

for i in classes:
    imageList = os.listdir(path + "\\" + i)
    for j in imageList:
        img = cv2.imread(path + "\\" + i + "\\" + j)
        img = cv2.resize(img, (32,32))
        images.append(img)
        outputs.append(i)
        
#%%
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

lb = LabelEncoder()
x = np.array(images)
outputs = lb.fit_transform(outputs)
y = to_categorical(outputs, num_classes=numberOfClasses)
#%%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

#%%
def preProcessImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

x_train = np.array(list(map(preProcessImage, x_train)))
x_test = np.array(list(map(preProcessImage, x_test)))

#%%
x_train = x_train.reshape(-1,32,32,1)
x_test = x_test.reshape(-1,32,32,1)
#%%
from keras.preprocessing.image import ImageDataGenerator
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.1,
                            rotation_range=10)

dataGen.fit(x_train)
#%%
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.layers import MaxPool2D
from keras import regularizers
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax

model = Sequential()
model.add(Conv2D(input_shape = (32,32,1), filters = 20, kernel_size = (5,5), activation = "relu", padding = "same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 40, kernel_size = (3,3), activation = "relu", padding = "same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=80, kernel_size=(3,3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=1024, activation = "relu"))
model.add(Dense(units=512, activation = "relu"))
model.add(Dense(units=7, activation = "softmax"))

model.compile(loss = "categorical_crossentropy", optimizer=("Adam"), metrics = ["accuracy"])

history = model.fit(x_train, y_train, epochs = 15,)

#%%
model.save("model.model")
