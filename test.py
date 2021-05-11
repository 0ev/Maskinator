from tensorflow.keras.models import model_from_json
from PIL import Image
import cv2
import numpy as np


with open("model.json", "r") as fp:
    model_test = model_from_json(fp.read())

model_test.load_weights("final.h5")

while True:
    inp = input("file name : ")

    img_array = np.array(Image.open(inp))

    img_array = cv2.resize(img_array, dsize=(140,160), interpolation=cv2.INTER_LINEAR)

    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    result = model_test.predict(img_array.reshape(1,160,140,1))

    print(result)