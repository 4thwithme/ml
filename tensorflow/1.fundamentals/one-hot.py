import tensorflow as tf

some_list = [3, 2, 7, 9]
some_words_list = [1, 2, 3, 4]

res_1 = tf.one_hot(some_list, depth=4, dtype=tf.int32)
res_1_2 = tf.one_hot(some_list, depth=2, dtype=tf.int32)

rez_2 = tf.one_hot(some_words_list, depth=4, dtype=tf.int32, axis=-1)

print(res_1)
print(res_1_2)
print(rez_2)
