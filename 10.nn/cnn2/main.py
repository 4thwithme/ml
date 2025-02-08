import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)

trainig_gen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

trainig_set = trainig_gen.flow_from_directory(
    "./dataset/train/", target_size=(224, 224), batch_size=32, class_mode="categorical"
)

testing_gen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

testing_set = testing_gen.flow_from_directory(
    "./dataset/test/", target_size=(224, 224), batch_size=32, class_mode="categorical"
)


cnn = tf.keras.models.Sequential()

cnn.add(
    tf.keras.layers.Conv2D(
        filters=32, kernel_size=4, activation="relu", input_shape=[224, 224, 3]
    )
)
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=4, activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation="relu"))
cnn.add(tf.keras.layers.Dense(units=15, activation="softmax"))

cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
cnn.fit(x=trainig_set, validation_data=testing_set, epochs=50)

cnn.save("model.h5")
