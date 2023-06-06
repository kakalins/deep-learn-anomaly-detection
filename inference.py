import os
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
from stattistic_utils import Statistics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)

# Função que baixa as imagens e cria um numpy para cada uma delas
def load_images(image_folder):
    images = []
    for filename in os.listdir(image_folder):
        if filename.endswith('NAMB', 5, 9) or filename.endswith('NENG', 5, 9):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path)
            images.append(np.array(image))
    return np.array(images)

# Path do model checkpoint para baixar o modelo
checkpoint_filepath = 'data\\models\\m1\\'      # Modelo do motor 1
#checkpoint_filepath = 'data\\models\\m2\\'      # Modelo do motor 2
#checkpoint_filepath = 'data\\models\\m3\\'      # Modelo do motor 3
#checkpoint_filepath = 'data\\models\\m123\\'      # Modelo para os tres motores juntos

x_test = None
y_test = None

if ('m1' in checkpoint_filepath):
    print('Modelo para o motor 1')
    # Baixa os dados de teste para o modelo 1
    valid_normal = load_images('data/images/test/M1/normal/M1N4/')
    valid_anomaly = load_images('data/images/test/M1/anomaly/M1A2/')
    valid_normal = valid_normal.T
    valid_normal = valid_normal[0].T
    valid_anomaly = valid_anomaly.T
    valid_anomaly = valid_anomaly[0].T

    print("Valid normal shape:", valid_normal.shape)

    x_test = np.concatenate((valid_normal, valid_anomaly), axis=0)

    print("Forma do conjunto de validação: ", x_test.shape)

    y_test = np.concatenate((np.zeros(valid_normal.shape[0], dtype=int), np.ones(valid_anomaly.shape[0], dtype=int)))
    print("Forma dos labels de validação: ", y_test.shape)

elif ('m2' in checkpoint_filepath):
    print('Modelo para o motor 2')
    valid_normal = load_images('data/images/test/M2/normal/M2N4/')
    valid_anomaly = load_images('data/images/test/M2/anomaly/M2A2/')

    valid_normal = valid_normal.T
    valid_normal = valid_normal[0].T
    valid_anomaly = valid_anomaly.T
    valid_anomaly = valid_anomaly[0].T

    print("Valid normal shape:", valid_normal.shape)
    print("Valid anomaly shape:", valid_anomaly.shape)
    
    x_test = np.concatenate((valid_normal, valid_anomaly), axis=0)

    print("Forma do conjunto de validação: ", x_test.shape)

    y_test = np.concatenate((np.zeros(valid_normal.shape[0], dtype=int), np.ones(valid_anomaly.shape[0], dtype=int)))
    print("Forma dos labels de validação: ", y_test.shape)

elif ('m3' in checkpoint_filepath):
    print('Modelo para o motor 3')
    valid_normal = load_images('data/images/test/M3/normal/M3N4/')
    valid_anomaly = load_images('data/images/test/M3/anomaly/M3A3/')
    valid_normal = valid_normal.T
    valid_normal = valid_normal[0].T
    valid_anomaly = valid_anomaly.T
    valid_anomaly = valid_anomaly[0].T

    print("Valid normal shape:", valid_normal.shape)

    x_test = np.concatenate((valid_normal, valid_anomaly), axis=0)

    print("Forma do conjunto de validação: ", x_test.shape)

    y_test = np.concatenate((np.zeros(valid_normal.shape[0], dtype=int), np.ones(valid_anomaly.shape[0], dtype=int)))
    print("Forma dos labels de validação: ", y_test.shape)
else:
    print('Modelo para o motor 123')
    valid_normal_m1 = load_images('data/images/test/M1/normal/M1N4/')
    valid_anomaly_m1 = load_images('data/images/test/M1/anomaly/M1A2/')

    valid_normal_m2 = load_images('data/images/test/M2/normal/M2N4/')
    valid_anomaly_m2 = load_images('data/images/test/M2/anomaly/M2A2/')

    valid_normal_m3 = load_images('data/images/test/M3/normal/M3N4/')
    valid_anomaly_m3 = load_images('data/images/test/M3/anomaly/M3A3/')

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

    x_valid_normal = np.concatenate((valid_normal_m1, valid_normal_m3, valid_normal_m3), axis=0)
    x_valid_anomaly = np.concatenate((valid_anomaly_m1, valid_anomaly_m3, valid_anomaly_m3), axis=0)
    x_test = np.concatenate((x_valid_normal, x_valid_anomaly), axis=0)
    print("Forma do conjunto de validação: ", x_test.shape)

    y_test = np.concatenate((np.zeros(x_valid_normal.shape[0], dtype=int), np.ones(x_valid_anomaly.shape[0], dtype=int)))
    print("Forma dos labels de validação: ", y_test.shape)

x_test = x_test / 255.0


models = os.listdir(checkpoint_filepath)
print(f"Modelos: {len(models)} {models}")
model = tf.keras.models.load_model(checkpoint_filepath + models[len(models) - 1])

if model == None:
    print("Erro ao baixar modelo")

y_pred = np.argmax(model.predict(x_test), axis=1)

stat = Statistics()

metrics = stat.calc_metrics(y_test, y_pred)
stat.print_metrics(metrics)
stat.get_confusion_matrix(y_test, y_pred)
stat.show_roc_curve(y_test, y_pred)
