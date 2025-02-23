import tensorflow as tf

some_list = [3, 2, 7, 9]


print(tf.one_hot(some_list, depth=4, dtype=tf.int32))
print(tf.one_hot(some_list, depth=5, dtype=tf.int32))
