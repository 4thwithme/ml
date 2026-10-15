import tensorflow as tf

H = tf.range(1, 10)
print(H)

H = tf.square(H)
print(H)

H = tf.math.sqrt(tf.cast(H, dtype=tf.float32))
print(H)

H = tf.math.log(tf.cast(H, dtype=tf.float32))
print(H)
