import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
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
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

# Function that builds the model with its hyperparameters
def model_builder(hp):
    model = Sequential()
    # Tune the number of units and the kernel size in the convolution layer
    hp_units = hp.Int('units', min_value = 16, max_value=128, step=16)
    hp_kernel_size = hp.Int('units', min_value=3, max_value=9, step=2)
    model.add(Conv2D(hp_units, (hp_kernel_size, hp_kernel_size), strides=1, padding="same", activation="relu", 
                    input_shape=(99, 151, 1)))
    
    model.add(BatchNormalization(trainable = False))
    model.add(MaxPool2D((2, 2), strides=2, padding="same"))
    model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
    model.add(BatchNormalization(trainable = False))
    model.add(MaxPool2D((2, 2), strides=2, padding="same"))
    model.add(Flatten())
    model.add(Dense(units=64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=num_classes, activation="softmax"))
    opt = Adam(learning_rate = 0.001)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
    return model

def load_images(image_folder):
    images = []
    for filename in os.listdir(image_folder):
        if filename.endswith('NAMB', 5, 9) or filename.endswith('NENG', 5, 9):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path)
            images.append(np.array(image))
    return np.array(images)

# Adicionado mecanismo de GPU para treinar mais rápido 
#tf.debugging.set_log_device_placement(True)


# Carregar as imagens e organizá-las em um tensor
system = os.name
if 'nt' in system:
    train_normal = load_images('data\\images\\validation\\M1\\normal\\M1N5\\')
    train_normal = np.append(train_normal, load_images('data\\images\\validation\\M1\\normal\\M1N6\\'), axis=0)
    train_anomaly = load_images('data\\images\\validation\\M1\\anomaly\\M1A3\\')
    train_anomaly = np.append(train_anomaly, load_images('data\\images\\validation\\M1\\anomaly\\M1A4\\'), axis=0)
    valid_normal = load_images('data\\images\\test\\M1\\normal\\M1N4\\')
    valid_anomaly = load_images('data\\images\\test\\M1\\anomaly\\M1A2\\')
else:
    train_normal = load_images('data/images/validation/M1/normal/M1N5/')
    train_normal = np.append(train_normal, load_images('data/images/validation/M1/normal/M1N6/'), axis=0)
    train_anomaly = load_images('data/images/validation/M1/anomaly/M1A3/')
    train_anomaly = np.append(train_anomaly, load_images('data/images/validation/M1/anomaly/M1A4/'), axis=0)
    valid_normal = load_images('data/images/test/M1/normal/M1N4/')
    valid_anomaly = load_images('data/images/test/M1/anomaly/M1A2/')

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

# Model Callback to save the model
checkpoint_filepath = 'data\\models\\m1\\'
my_callback = [
    keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath + 'm1_model.h5', monitor = "val_loss", save_best_only = True)
]

tuner = kt.Hyperband(model_builder, objective='val_accuracy', max_epochs=10, factor = 3, directory=checkpoint_filepath, project_name='m1_hyper')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# Build the model with the optimal hyperparameters and train it on the data for 50 epochs

""" model = Sequential()
model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu", 
                 input_shape=(99, 151, 1)))
model.add(BatchNormalization(trainable = False))
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
model.add(Dense(units=num_classes, activation="softmax")) """

#model.summary()

opt = Adam(learning_rate = 0.001)
#model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])

tuner.search(x_train, y_train, batch_size = 5, epochs=10, verbose=1, validation_data=(x_valid, y_valid), callbacks=[stop_early])
# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')}.
""")

model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, batch_size = 5, epochs=50, verbose=1, validation_data=(x_valid, y_valid), callbacks=my_callback)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


