import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
from sklearn.metrics import confusion_matrix
from PIL import Image
import os
from keras.datasets import mnist


# Load Imagens from dataset de motores
def load_images(image_folder):
    images = []
    for filename in os.listdir(image_folder):
        if filename.endswith('NAMB', 5, 9) or filename.endswith('NENG', 5, 9):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path)
            images.append(np.array(image))
    return np.array(images)

# Function to return predictions
def predict(_model, _x_test):
    results = []
    for i in range(len(_x_test)):
        result = tf.argmax(_model.predict(tf.expand_dims(_x_test[i], 0)), axis=1)
        results.append(result)
        #print(result.numpy(), y_test[i])
    return np.array(results).reshape(len(results),)

# Carregar as imagens e organizá-las em um tensor
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
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_valid = keras.utils.to_categorical(y_valid, num_classes)
x_train = x_train / 255
x_valid = x_valid / 255
print(x_train.min(), x_train.max())
print(x_valid.min(), x_valid.max())

# Model Callback to save the model
checkpoint_filepath = 'data\\models\\lstm\\m1\\'
my_callback = [
    keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath + 'm1_model.{epoch: 02d}--{val_loss: .2f}.h5', monitor = "val_loss", save_best_only = True)
]

""" #Importing the data
(x_train, y_train),(x_test, y_test) = mnist.load_data() 

#Normalizing the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0 """

# Create a model in Tensorflow for an LSTM
model = keras.Sequential()
model.add(layers.LSTM(128, input_shape = (x_train.shape[1:]), return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(128))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(
    loss='binary_crossentropy',
    optimizer="adam",
    metrics=["accuracy"]
)

# Print Summary of the model
print("--"*10)
print(model.summary())
print("--"*10)

model.fit(x_train,y_train, batch_size = 5, epochs=50, validation_data = (x_valid, y_valid), callbacks = [my_callback])