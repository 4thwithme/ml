import tensorflow as tf

t1 = tf.constant([[1, 1], [1, 1]])
t2 = tf.constant([[2, 2], [2, 2]])

print(t1 * t2)  # <--- element-wise multiplication
print(tf.linalg.matmul(t1, t2))  # <--- matrix multiplication

t3 = tf.constant([[1, 2, 5], [7, 2, 1], [3, 3, 3]])
t4 = tf.constant([[3, 5], [6, 7], [1, 8]], shape=(3, 2), dtype=tf.int32)

print(tf.linalg.matmul(t3, t4))  # <--- matrix multiplication
print(t3 @ t4)  # <--- matrix multiplication
# print(t4 @ t3)  # <--- error: incompatible shapes. Inner dimensions must match
print("--------------")
print(t4)
reshaped_t4 = tf.reshape(t4, shape=(2, 3))
print("reshaped", reshaped_t4)
print(tf.transpose(t4))
print("--------------")
print(reshaped_t4 @ t3)  # <--- matrix multiplication

print(tf.matmul(reshaped_t4, t3))  # <--- matrix multiplication
