import tensorflow as tf

print(tf.__version__)

# create variable

var = tf.Variable(7)  # 0D tensor
print("var", var)
var.assign(10)
print("var", var)

var2 = tf.Variable([1, 2, 3, 4, 5])  # 1D tensor
print("var2", var2)
var2.assign([10, 20, 30, 40, 50])
print("var2", var)
