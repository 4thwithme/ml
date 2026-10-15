import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

print(tf.__version__)

print("Hello World!")

# Check for GPU availability
gpu_devices_list = tf.config.list_physical_devices("GPU")
print("GPU available:", gpu_devices_list)

# Create sample data
X = tf.range(0, 100, 1, dtype=tf.float32)
y = X * 1.2 + 10

# Build and train model
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Input(shape=(1,)))
ann.add(tf.keras.layers.Dense(units=180, activation="relu"))
ann.add(tf.keras.layers.Dense(units=1))

# Compile the model
ann.compile(
    optimizer="adam",
    loss="mae",
    metrics=["mae"],
)

# Train the model
ann.fit(X, y, batch_size=12, epochs=200)

# Make predictions - reshape input to 2D
prediction_input = tf.constant([[14.2]])  # Note the extra brackets for 2D shape
print(ann.predict(prediction_input))


# Plot the data
plt.scatter(X, y)
plt.plot(X, ann.predict(np.array(X).reshape(-1, 1)), color="red")
plt.show()
