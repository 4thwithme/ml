import numpy as np

import tensorflow as tf

J = tf.constant(np.array([[1, 2], [3, 4]]))
K = tf.constant(np.array([[5, 6], [7, 8]]))

print(J)
print(K)

arr = np.array(J)
print("np", arr, type(arr))

np_arr = J.numpy()
print("np_arr", np_arr, type(np_arr))
