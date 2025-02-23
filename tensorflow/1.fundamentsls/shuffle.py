import tensorflow as tf

not_shuffled = tf.constant([[10, 20], [30, 40], [50, 60]])

shuffled = tf.random.shuffle(not_shuffled)
print(shuffled)
