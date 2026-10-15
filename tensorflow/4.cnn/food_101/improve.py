import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Load the previously trained model
print("Loading the existing model...")
model = tf.keras.models.load_model("./food101_model.h5")

# Print the current model architecture
model.summary()

# Load the same data
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

# Enhanced data augmentation for continued training
training_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,  # More rotation
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,  # More zoom
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Brightness variation
    fill_mode="nearest",
)

testing_generator = ImageDataGenerator(
    rescale=1.0 / 255
)  # No augmentation for test data

# Setup data generators
training_set = training_generator.flow_from_dataframe(
    dataframe=train_list,
    directory="./data/images/",
    x_col="ImagePath",
    y_col="Label",
    target_size=(340, 340),  # Keep same size as original model
    batch_size=32,
    class_mode="categorical",
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

# Improved callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=20,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.1,  # Stronger reduction
        patience=10,
        min_lr=1e-6,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "food101_improved_model.h5", monitor="val_accuracy", save_best_only=True
    ),
]

# Recompile with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc"),
    ],
)

# Continue training with improved parameters
print("Fine-tuning the existing model...")
model.fit(
    x=training_set,
    validation_data=testing_set,
    epochs=30,  # Continue training for more epochs
    callbacks=callbacks,
    batch_size=32,
    validation_steps=len(testing_set),
    steps_per_epoch=len(training_set),
)

# Evaluate the improved model
print("Evaluating the improved model...")
evaluation = model.evaluate(testing_set)
print(f"Test Loss: {evaluation[0]}")
print(f"Test Accuracy: {evaluation[1]}")
print(f"Test Top-5 Accuracy: {evaluation[2]}")

# Save the improved model
model.save("food101_model_improved_final.h5")
print("Training complete. Improved model saved as food101_model_improved_final.h5")
