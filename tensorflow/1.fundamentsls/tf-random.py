import tensorflow as tf

g = tf.random.Generator.from_seed(42)
g = g.normal(shape=(2, 3))

g2 = tf.random.Generator.from_seed(42)
g2 = g2.normal(shape=(2, 3))
print(g, g2, g == g2)
