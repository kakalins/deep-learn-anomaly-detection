import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
from sklearn.metrics import confusion_matrix
from PIL import Image
import os
from keras.datasets import mnist
from stattistic_utils import Statistics


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
def parse_prediction(y_pred, threshold):
    y_out = np.zeros(shape=y_pred.shape, dtype=float)
    for i, prob in enumerate(y_pred):
        if prob > threshold:
            y_out[i] = 1.
    return y_out

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
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_valid = keras.utils.to_categorical(y_valid, num_classes)
x_train = x_train / 255
x_valid = x_valid / 255
print(x_train.min(), x_train.max())
print(x_valid.min(), x_valid.max())

# Model Callback to save the model
checkpoint_filepath = 'data\\models\\lstm\\m2\\'
my_callback = [
    keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath + 'm2_model.h5', monitor = "val_loss", save_best_only = True)
]

# Create a model in Tensorflow for an LSTM
model = keras.Sequential()
model.add(layers.LSTM(128, input_shape = (x_train.shape[1:]), return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(128))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
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

model.fit(x_train,y_train, batch_size = 5, epochs=20, validation_data = (x_valid, y_valid), callbacks = [my_callback])

# Evaluate the model
models = os.listdir(checkpoint_filepath)
model_name = ""
for name in models:
    if ".h5" in name:
        model_name = name

print(f"Modelos: {len(models)} {models}")
model_pred = tf.keras.models.load_model(checkpoint_filepath + model_name)
stat = Statistics()

if model_pred == None:
    print("Erro ao baixar modelo")
else:
    y_prediction = parse_prediction(model_pred.predict(x_valid), 0.5)
    metrics = stat.calc_metrics(y_valid, y_prediction)
    stat.print_metrics(metrics)
    stat.save_metrics(metrics, checkpoint_filepath, "result")





