import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from sklearn.metrics import confusion_matrix
import os
from stattistic_utils import Statistics

sequence_leght = 480
path = 'C:\\Users\\pves\\Documents\\Voxar_Contexto\\Workspace\\Deep_Learning\\Projeto\\deep-learn-anomaly-detection\\data\\m1.csv'
# Function to return predictions
def parse_prediction(y_pred, threshold):
    y_out = np.zeros(shape=y_pred.shape, dtype=float)
    for i, prob in enumerate(y_pred):
        if prob > threshold:
            y_out[i] = 1.
    return y_out

def generate_data(X, y, sequence_length = 10, step = 1):
    X_local = []
    y_local = []
    for start in range(0, len(X) - sequence_length, step):
        end = start + sequence_length
        X_local.append(X[start:end])
        y_local.append(y[end-1])
    return np.array(X_local), np.array(y_local)

m1_dataframe = pd.read_csv(path, index_col=0)
anomaly = m1_dataframe[m1_dataframe['class'] == 1]
normal = m1_dataframe[m1_dataframe['class'] == 0]
print("Inicio anomalia {}".format(anomaly.index[0]))
print("Inicio Normal {}".format(normal.index[0]))
print("**"*50)

(x_anomally, y_anomally) = generate_data(anomaly.iloc[:,:-1].values, anomaly['class'].values, 480)
(x_normal, y_normal) = generate_data(normal.iloc[:,:-1].values, normal['class'].values, 480)
print("X_anomally shape and y anomally shape = {a}, {b}".format(a = x_anomally.shape, b = y_anomally.shape))
print("X_normal shape and y normal shape = {a}, {b}".format(a = x_normal.shape, b = y_normal.shape))
print("**"*50)

# Separating the test and training sets
training = 0.7
validation = 0.1
test = 0.2

############# - X Partitioning- ######################################################################################
end_a = int(len(x_anomally) * training)
end_n = int(len(x_normal) * training)

x_train_anomally = x_anomally[: end_a]
x_train_normal = x_normal[: end_n]

x_validation_anomally = x_anomally[end_a : end_a + int(len(x_anomally) * validation)]
x_validation_normal = x_normal[end_n : end_n + int(len(x_normal) * validation)]
end_n = end_n + int(len(x_normal) * validation)
end_a = end_a + int(len(x_anomally) * validation)

x_test_anomally = x_anomally[end_a :]
x_test_normal = x_normal[end_n  : ]


############# - Y Partitioning- ######################################################################################

end_a = int(len(x_anomally) * training)
end_n = int(len(x_normal) * training)

y_train_anomally = y_anomally[: end_a]
y_train_normal = y_normal[: end_n]

y_validation_anomally = y_anomally[end_a : end_a + int(len(x_anomally) * validation)]
y_validation_normal = y_normal[end_n : end_n + int(len(x_normal) * validation)]
end_n = end_n + int(len(x_normal) * validation)
end_a = end_a + int(len(x_anomally) * validation)

y_test_anomally = y_anomally[end_a :]
y_test_normal = y_normal[end_n  : ]


print("Train anomally shape = {a}, {b}".format(a = x_train_anomally.shape, b = y_train_anomally.shape))
print("Validation anomally shape = {a}, {b}".format(a = x_validation_anomally.shape, b = y_validation_anomally.shape))
print("Test anomally shape = {a}, {b}".format(a = x_test_anomally.shape, b = y_test_anomally.shape))
print('**'*50)
print("Train Normal shape = {a}, {b}".format(a = x_train_normal.shape, b = y_train_normal.shape))
print("Validation Normal shape = {a}, {b}".format(a = x_validation_normal.shape, b = y_validation_normal.shape))
print("Test Normal Shape = {a}, {b}".format(a = x_test_normal.shape, b = y_test_normal.shape))

x_train = np.concatenate([x_train_anomally, x_train_normal])
y_train = np.concatenate([y_train_anomally, y_train_normal])

x_valid = np.concatenate([x_validation_anomally, x_validation_normal])
y_valid = np.concatenate([y_validation_anomally, y_validation_normal])

x_test = np.concatenate([x_test_anomally, x_test_normal])
y_test = np.concatenate([y_test_anomally, y_test_normal])

print("**"*50)
print("Train shape = {a}, {b}".format(a = x_train.shape, b = y_train.shape))
print("Validation shape = {a}, {b}".format(a = x_valid.shape, b = y_valid.shape))
print("Test shape = {a}, {b}".format(a = x_test.shape, b = y_test.shape))
print("**"*50)
# Model Callback to save the model
checkpoint_filepath = 'data\\models\\lstm\\m1\\'
# Experiments
engine_name = "m1"

# Results path
results_path = "C:\\Users\\pves\\Documents\\Voxar_Contexto\\Workspace\\Deep_Learning\\Projeto\\deep-learn-anomaly-detection\\data\\results\\lstm\\"
my_callback = [
    keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath + 'm1_model.h5', monitor = "val_accuracy", save_best_only = True)
]

# Combinações de HP


# Create a model in Tensorflow for an LSTM
model = keras.Sequential()
model.add(layers.LSTM(64, input_shape = (x_train.shape[1],x_train.shape[2]), return_sequences=True))
model.add(layers.Dropout(0.3))
model.add(layers.LSTM(32))
#model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dropout(0.2))
#model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compiling the model
#adam = keras.optimizers.Adam(learning_rate = 0.00001)
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=["accuracy"]
)

print(model.summary())


model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_valid, y_valid), verbose=1, callbacks = my_callback)

models = os.listdir(checkpoint_filepath)
print(f"Modelos: {len(models)} {models}")
model = tf.keras.models.load_model(checkpoint_filepath + models[len(models) - 1])

if model == None:
    print("Erro ao baixar modelo")

y_pred = model.predict(x_test)
y_pred = parse_prediction(y_pred, 0.5)

stat = Statistics()

metrics = stat.calc_metrics(y_test, y_pred)
stat.print_metrics(metrics)
stat.get_confusion_matrix(y_test, y_pred)
stat.show_roc_curve(y_test, y_pred)
stat.save_metrics(metrics, results_path, engine_name)