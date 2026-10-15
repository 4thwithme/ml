import tensorflow as tf

t4 = tf.zeros([2, 3, 3, 4])
print(t4)
print("t4.shape", t4.shape)
print("t4.dtype", t4.dtype)
print("t4.numpy", t4.numpy())
print("t4.ndim", t4.ndim)
print("t4.device", t4.device)
print("Element along axis 0", t4.shape[0])
print("Element along last axis", t4.shape[-1])
print("Total number of elements", tf.size(t4).numpy())

sub_tensor = t4[:2, :2, :2, :2]
print(sub_tensor)
