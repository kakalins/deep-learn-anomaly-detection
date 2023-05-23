from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

#ImgDataGen = keras.preprocessing.image.ImageDataGenerator
base_model = keras.applications.VGG16(
    weights='imagenet',
    input_shape=(99, 151, 3),
    include_top=False)

# Freeze base model
base_model.trainable = False

# Create inputs with correct shape
inputs = keras.Input(shape=(99, 151, 3))

x = base_model(inputs, training=False)

# Add pooling layer or flatten layer
x = keras.layers.GlobalAveragePooling2D()(x)

# Add final dense layer
outputs = keras.layers.Dense(2, activation = 'softmax')(x)
#outputs = keras.layers.Dense(1)(x)

# Combine inputs and outputs to create model
model = keras.Model(inputs, outputs)

model.summary()

model.compile(loss = 'categorical_crossentropy' , metrics = ['accuracy'])

datagen_train = ImageDataGenerator(samplewise_center=True, rescale=1/255)
datagen_valid = ImageDataGenerator(samplewise_center=True, rescale=1/255)

train_it = datagen_train.flow_from_directory(
    "data/images/training/M1",
    target_size=(99,151),
    color_mode="rgb",
    class_mode="categorical",
)

valid_it = datagen_valid.flow_from_directory(
    "data/images/validation/M1",
    target_size=(99,151),
    color_mode="rgb",
    class_mode="categorical",
)

model.fit(train_it,
          validation_data=valid_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=valid_it.samples/valid_it.batch_size,
          epochs=10)