# How to check if GPU is enabled for TensorFlow?

import tensorflow as tf

# Check if GPU is enabled for TensorFlow
is_gpu_available = tf.test.is_gpu_available()
print("GPU available:", is_gpu_available)

# will show gpu if u installed `python -m pip install tensorflow-metal`
devices_list = tf.config.list_physical_devices()
print("Devices list:", devices_list)
