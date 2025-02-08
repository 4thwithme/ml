import numpy as np
import tensorflow as tf
from keras.preprocessing import image

# load model
cnn = tf.keras.models.load_model("model.h5")

test_image = image.image_utils.load_img(
    "./dataset/single_prediction/natali2.png", target_size=(128, 128)
)

test_image = image.image_utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)


if result[0][0] == 1:
    prediction = "dog"
else:
    prediction = "cat"

print(prediction)
