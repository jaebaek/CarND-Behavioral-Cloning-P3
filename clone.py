import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

lines = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    steering_center = float(line[3])

    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # read in images from center, left and right cameras
    img_center = cv2.imread(line[0])
    img_left = cv2.imread(line[1])
    img_right = cv2.imread(line[2])

    # add images and angles to data set
    images.extend([img_center, img_left, img_right])
    measurements.extend([steering_center, steering_left, steering_right])

    # Flipped image
    flipped_img_center, flipped_img_left, flipped_img_right = np.fliplr(img_center), np.fliplr(img_left), np.fliplr(img_right)
    flipped_steering_center, flipped_steering_left, flipped_steering_right = [-steering_center, -steering_left, -steering_right]

    # add flipped images and angles to data set
    images.extend([flipped_img_center, flipped_img_left, flipped_img_right])
    measurements.extend([flipped_steering_center, flipped_steering_left, flipped_steering_right])

X_train = np.array(images)
y_train = np.array(measurements)

print(X_train.shape, y_train.shape)
print(y_train.max(), y_train.min())

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

# Layer 1
model.add(Convolution2D(24, 5, 5))
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 2
model.add(Convolution2D(36, 5, 5))
model.add(Activation('relu'))
model.add(Dropout(0.7))
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 3
model.add(Convolution2D(48, 5, 5))
model.add(Activation('relu'))
model.add(Dropout(0.7))
model.add(MaxPooling2D(pool_size=(2,2)))

## Layer 4
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Dropout(0.7))

# Layer 5
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.7))

# Flatten
model.add(Flatten())

# Layer 6
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Layer 7
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Layer 8
model.add(Dense(10))
model.add(Dropout(0.5))

# Layer 9
model.add(Dense(1))

model.compile('adam', 'mse')
model.fit(X_train, y_train, nb_epoch=50, validation_split=0.2, shuffle=True)
model.save('model.h5')
