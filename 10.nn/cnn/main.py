import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)

# Feature Scaling
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)
trainig_set = train_datagen.flow_from_directory(
    "./dataset/training_set/",
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_set = test_datagen.flow_from_directory(
    "./dataset/test_set/",
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
)

# CNN

cnn = tf.keras.models.Sequential()
# first convolutional layer
cnn.add(
    tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, activation="relu", input_shape=[128, 128, 3]
    )
)

cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))

# second convolutional layer

cnn.add(
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation="relu",
    )
)

cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))

# Flattening step
cnn.add(tf.keras.layers.Flatten())

# add input and 1st hidden layer
cnn.add(tf.keras.layers.Dense(units=168, activation="relu"))
# add output layer
cnn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# compile CNN
cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

cnn.fit(x=trainig_set, validation_data=test_set, epochs=25)

# Making a single prediction
test_image = image.image_utils.load_img(
    "./dataset/single_prediction/cat_or_dog.jpg", target_size=(128, 128)
)

test_image = image.image_utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)

print(result)

trainig_set.class_indices
if result[0][0] == 1:
    prediction = "dog"
else:
    prediction = "cat"

print(prediction)

# save model

cnn.save("model.h5")
