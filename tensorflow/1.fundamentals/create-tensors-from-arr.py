import numpy as np

import tensorflow as tf

ones = tf.ones([3, 5])
print(ones)

zeroes = tf.zeros([3, 4])
print(zeroes)

arr_1 = np.arange(1, 25, dtype=np.int32)

t1 = tf.convert_to_tensor(arr_1, dtype=tf.int32)
print(t1)
arr_2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
t2 = tf.convert_to_tensor(arr_2, dtype=tf.float16)
print(t2)

t3 = tf.constant(arr_1, shape=(2, 3, 4))
print(t3)
