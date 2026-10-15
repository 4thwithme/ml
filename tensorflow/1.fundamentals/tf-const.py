import tensorflow as tf

print(tf.__version__)

# create constant

scalar = tf.constant(7)  # 0D tensor
vector = tf.constant([1, 2, 3, 4, 5])  # 1D tensor
matrix = tf.constant([[1, 2, 3], [4, 5, 6]])  # 2D tensor
tensor = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])  # 3D tensor

print("scalar", scalar, scalar.ndim)
print("vector", vector, vector.ndim)
print("matrix", matrix, matrix.ndim)
print("tensor", tensor, tensor.ndim)

# float precision. f16 takes less memory than f32

float_matrix = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float16)
print("float_matrix", float_matrix, float_matrix.ndim)
