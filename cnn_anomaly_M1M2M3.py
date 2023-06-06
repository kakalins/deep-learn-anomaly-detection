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


# Carregar as imagens e organizá-las em um tensor para o motor 1
train_normal_m1 = load_images('data/images/validation/M1/normal/M1N5/')
train_normal_m1 = np.append(train_normal_m1, load_images('data/images/validation/M1/normal/M1N6/'), axis=0)
train_anomaly_m1 = load_images('data/images/validation/M1/anomaly/M1A3/')
train_anomaly_m1 = np.append(train_anomaly_m1, load_images('data/images/validation/M1/anomaly/M1A4/'), axis=0)
valid_normal_m1 = load_images('data/images/test/M1/normal/M1N4/')
valid_anomaly_m1 = load_images('data/images/test/M1/anomaly/M1A2/')

# Carregar as imagens e organizá-las em um tensor para o motor 2
train_normal_m2 = load_images('data/images/validation/M2/normal/M2N5/')
train_normal_m2 = np.append(train_normal_m2, load_images('data/images/validation/M2/normal/M2N6/'), axis=0)
train_anomaly_m2 = load_images('data/images/validation/M2/anomaly/M2A3/')
train_anomaly_m2 = np.append(train_anomaly_m2, load_images('data/images/validation/M2/anomaly/M2A4/'), axis=0)
valid_normal_m2 = load_images('data/images/test/M2/normal/M2N4/')
valid_anomaly_m2 = load_images('data/images/test/M2/anomaly/M2A2/')

# Carregar as imagens em um tensor para o motor 3
train_normal_m3 = load_images('data/images/validation/M3/normal/M3N2/')
train_normal_m3 = np.append(train_normal_m3, load_images('data/images/validation/M3/normal/M3N3/'), axis=0)
train_anomaly_m3 = load_images('data/images/validation/M3/anomaly/M3A1/')
train_anomaly_m3 = np.append(train_anomaly_m3, load_images('data/images/validation/M3/anomaly/M3A2/'), axis=0)
valid_normal_m3 = load_images('data/images/test/M3/normal/M3N4/')
valid_anomaly_m3 = load_images('data/images/test/M3/anomaly/M3A3/')

# verificando a forma do tensor
train_normal_m1 = train_normal_m1.T
train_normal_m1 = train_normal_m1[0].T

train_normal_m2 = train_normal_m2.T
train_normal_m2 = train_normal_m2[0].T

train_normal_m3 = train_normal_m3.T
train_normal_m3 = train_normal_m3[0].T

#print("Train normal shape:", train_normal.shape)

train_anomaly_m1 = train_anomaly_m1.T
train_anomaly_m1 = train_anomaly_m1[0].T

train_anomaly_m2 = train_anomaly_m2.T
train_anomaly_m2 = train_anomaly_m2[0].T

train_anomaly_m3 = train_anomaly_m3.T
train_anomaly_m3 = train_anomaly_m3[0].T
#print("Train anomaly shape:", train_anomaly.shape)

valid_normal_m1 = valid_normal_m1.T
valid_normal_m1 = valid_normal_m1[0].T

valid_normal_m2 = valid_normal_m2.T
valid_normal_m2 = valid_normal_m2[0].T

valid_normal_m3 = valid_normal_m3.T
valid_normal_m3 = valid_normal_m3[0].T
#print("Valid normal shape:", valid_normal.shape)

valid_anomaly_m1 = valid_anomaly_m1.T
valid_anomaly_m1 = valid_anomaly_m1[0].T

valid_anomaly_m2 = valid_anomaly_m2.T
valid_anomaly_m2 = valid_anomaly_m2[0].T

valid_anomaly_m3 = valid_anomaly_m3.T
valid_anomaly_m3 = valid_anomaly_m3[0].T
#print("Valid anomaly shape:", valid_anomaly.shape)



x_train_normal = np.concatenate((train_normal_m1, train_normal_m2, train_normal_m3), axis=0)
print("Forma do conjunto de treinamento: ", x_train_normal.shape)
x_train_anomaly = np.concatenate((train_anomaly_m1, train_anomaly_m2, train_anomaly_m3), axis=0)
print("Forma do conjunto de treinamento: ", x_train_normal.shape)
x_train = np.concatenate((x_train_normal, x_train_anomaly), axis=0)

y_train = np.concatenate((np.zeros(x_train_normal.shape[0], dtype=int), np.ones(x_train_anomaly.shape[0], dtype=int)))
print("Forma dos labels de treinamento: ", y_train.shape)

x_valid_normal = np.concatenate((valid_normal_m1, valid_normal_m3, valid_normal_m3), axis=0)
x_valid_anomaly = np.concatenate((valid_anomaly_m1, valid_anomaly_m3, valid_anomaly_m3), axis=0)
x_valid = np.concatenate((x_valid_normal, x_valid_anomaly), axis=0)
print("Forma do conjunto de validação: ", x_valid.shape)

y_valid = np.concatenate((np.zeros(x_valid_normal.shape[0], dtype=int), np.ones(x_valid_anomaly.shape[0], dtype=int)))
print("Forma dos labels de validação: ", y_valid.shape)

num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
x_train = x_train / 255
x_valid = x_valid / 255
print(x_train.min(), x_train.max())
print(x_valid.min(), x_valid.max())

# Model Callback to save the model
checkpoint_filepath = 'data\\models\\m123\\'
my_callback = [
    keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath + 'm123_model.{epoch: 02d}--{val_loss: .2f}.h5', monitor = "val_loss", save_best_only = True)
]

model = Sequential()

model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu", 
                 input_shape=(99, 151, 1)))
#model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(32, (3, 3), strides=1, padding="same", activation="relu"))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
#model.add(Conv2D(16, (3, 3), strides=1, padding="same", activation="relu"))
#model.add(BatchNormalization())
#model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Flatten())
model.add(Dense(units=64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=num_classes, activation="softmax"))

model.summary()


model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

hist = model.fit(x_train, y_train, batch_size = 5, epochs=100, verbose=1, validation_data=(x_valid, y_valid), callbacks=my_callback)