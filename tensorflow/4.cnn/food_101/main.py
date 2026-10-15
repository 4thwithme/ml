import os

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

# Print TensorFlow version for debugging
print(f"TensorFlow version: {tf.__version__}")

# Check for GPU availability
is_gpu_used = tf.config.list_physical_devices("GPU")
print(f"Available GPUs: {is_gpu_used}")

classes_list = pd.read_csv(
    "./data/meta/classes.txt", header=None, delimiter="\t", names=["ClassName"]
)
labels_list = pd.read_csv(
    "./data/meta/labels.txt", header=None, delimiter="\t", names=["Label"]
)
test_list = pd.read_csv(
    "./data/meta/test.txt", header=None, delimiter="\t", names=["ImageID"]
)
train_list = pd.read_csv(
    "./data/meta/train.txt", header=None, delimiter="\t", names=["ImageID"]
)

train_list["Label"] = train_list["ImageID"].apply(lambda x: x.split("/")[0])
test_list["Label"] = test_list["ImageID"].apply(lambda x: x.split("/")[0])
train_list["ImagePath"] = train_list["ImageID"] + ".jpg"
test_list["ImagePath"] = test_list["ImageID"] + ".jpg"

print(classes_list)
print(labels_list)
print(test_list)
print(train_list)


training_generator = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

training_set = training_generator.flow_from_dataframe(
    dataframe=train_list,
    directory="./data/images/",
    x_col="ImagePath",
    y_col="Label",
    target_size=(340, 340),
    batch_size=32,
    class_mode="categorical",
)

testing_generator = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

testing_set = testing_generator.flow_from_dataframe(
    dataframe=test_list,
    directory="./data/images/",
    x_col="ImagePath",
    y_col="Label",
    target_size=(340, 340),
    batch_size=32,
    class_mode="categorical",
)


cnn = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, activation="relu", input_shape=[340, 340, 3]
        ),
        tf.keras.layers.Conv2D(filters=32, kernel_size=4, activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=3),
        tf.keras.layers.Conv2D(filters=16, kernel_size=4, activation="relu"),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=1010, activation="relu"),
        tf.keras.layers.Dense(units=101, activation="softmax"),
    ]
)

# Print model summary
cnn.summary()


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="accuracy", patience=40, restore_best_weights=True, 
    ),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="accuracy", factor=0.2, patience=40),
    tf.keras.callbacks.ModelCheckpoint("food101_best_model.h5", save_best_only=True),
]


cnn.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc"),
    ],
)
cnn.fit(
    x=training_set,
    validation_data=testing_set,
    epochs=50,
    callbacks=callbacks,
    batch_size=32,
    validation_steps=len(testing_set),
    steps_per_epoch=len(training_set),
)

cnn.save("food101_model.h5")
print("Training complete. Model saved as food101_model.h5")
