import numpy

import tensorflow as tf

t1 = tf.constant(numpy.random.randint(5, 100, size=(9, 5)))
print(t1)
print("Min", tf.reduce_min(t1))
print("Max", tf.reduce_max(t1))
print("Mean", tf.reduce_mean(t1))
print("Sum", tf.reduce_sum(t1))
print("Standard Deviation", tf.math.reduce_std(tf.cast(t1, dtype=tf.float32)))
print("Variance", tf.math.reduce_variance(tf.cast(t1, dtype=tf.float32)))

print("Index Of Max by Axes 0", tf.argmax(t1, 0))
print("Index Of Min by axes 0", tf.argmin(t1, 0))

print("Index Of Max by Axes 1", tf.argmax(t1, 1))
print("Index Of Min by axes 1", tf.argmin(t1, 1))


t2 = tf.constant(numpy.random.randint(5, 100, size=(1, 1, 1, 3, 1, 5)))
t2 = tf.squeeze(t2)
print(t2)
