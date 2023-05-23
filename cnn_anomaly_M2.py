import os
import numpy as np
from PIL import Image
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)


def load_images(image_folder):
    images = []
    for filename in os.listdir(image_folder):
        if filename.endswith('NAMB', 5, 9) or filename.endswith('NENG', 5, 9):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path)
            images.append(np.array(image))
    return np.array(images)


# Carregar as imagens e organizá-las em um tensor
train_normal = load_images('data/images/validation/M2/normal/M2N5/')
train_normal = np.append(train_normal, load_images('data/images/validation/M2/normal/M2N6/'), axis=0)
train_anomaly = load_images('data/images/validation/M2/anomaly/M2A3/')
train_anomaly = np.append(train_anomaly, load_images('data/images/validation/M2/anomaly/M2A4/'), axis=0)
valid_normal = load_images('data/images/test/M2/normal/M2N4/')
valid_anomaly = load_images('data/images/test/M2/anomaly/M2A2/')

# verificando a forma do tensor
train_normal = train_normal.T
train_normal = train_normal[0].T
print("Train normal shape:", train_normal.shape)

train_anomaly = train_anomaly.T
train_anomaly = train_anomaly[0].T
print("Train anomaly shape:", train_anomaly.shape)

valid_normal = valid_normal.T
valid_normal = valid_normal[0].T
print("Valid normal shape:", valid_normal.shape)

valid_anomaly = valid_anomaly.T
valid_anomaly = valid_anomaly[0].T
print("Valid anomaly shape:", valid_anomaly.shape)

x_train = np.concatenate((train_normal, train_anomaly), axis=0)
print("Forma do conjunto de treinamento: ", x_train.shape)

y_train = np.concatenate((np.zeros(train_normal.shape[0], dtype=int), np.ones(train_anomaly.shape[0], dtype=int)))
print("Forma dos labels de treinamento: ", y_train.shape)

x_valid = np.concatenate((valid_normal, valid_anomaly), axis=0)
print("Forma do conjunto de validação: ", x_valid.shape)

y_valid = np.concatenate((np.zeros(valid_normal.shape[0], dtype=int), np.ones(valid_anomaly.shape[0], dtype=int)))
print("Forma dos labels de validação: ", y_valid.shape)

num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
x_train = x_train / 255
x_valid = x_valid / 255
print(x_train.min(), x_train.max())
print(x_valid.min(), x_valid.max())

model = Sequential()

model.add(Conv2D(4, (3, 3), strides=1, padding="same", activation="relu", 
                 input_shape=(99, 151, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(8, (3, 3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(16, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Flatten())
model.add(Dense(units=16, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=num_classes, activation="softmax"))

model.summary()


model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_valid, y_valid))
