import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image

model = load_model("model.h5")

test_image = image.image_utils.load_img(
# put img here
    "./dataset/validation/porsche.png", target_size=(224, 224)
)

test_image = image.image_utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

print(result)

categories = [
    "Bean",
    "Bitter gourd",
    "Bottle gourd",
    "Brinjal",
    "Broccoli",
    "Cabbage",
    "Capsicum",
    "Carrot",
    "Cauliflower",
    "Cucumber",
    "Papaya",
    "Potato",
    "Pumpkin",
    "Radish",
    "Tomato",
]

print(categories[np.argmax(result)])
