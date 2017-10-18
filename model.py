import csv
import cv2
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout

# Load all the .csv input
lines = []
with open('C:/Users/joris/Desktop//driving_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Load all the images and corresponding steering angles
images = []
angles = []
for line in lines:
    img_path = line[0]
    filename = img_path.split('\\')[-1]
    curr_path = 'C:/Users/joris/Desktop/driving_data/IMG/' + filename
    img = cv2.imread(curr_path)
    images.append(img)
    
    angle = float(line[3])
    angles.append(angle)

aug_images, aug_angles = [], []
for image, angle in zip(images, angles):
    aug_images.append(image)
    aug_angles.append(angle)
    # aug_images.append(cv2.flip(image, 1))
    # aug_angles.append(-angle)

X_train = np.array(aug_images)
y_train = np.array(aug_angles)

# Create the neural network
model = Sequential()

# Normalize the input
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))

model.add(Convolution2D(96, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(64, (3, 3), strides=(2, 2), activation="relu"))
model.add(Flatten())
model.add(Dense(1000))
model.add(Dropout(0.5))
model.add(Dense(400))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(1))

# model = load_model('model.h5')

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)

model.save('model.h5')


