import requests
import cv2
import os
import numpy as np

def upload(name,img):
    cv2.imwrite(f'images/{name}.png',img)
    file = open(f'images/{name}.png','rb')
    data = {"file" : (str(name)+".png",file, 'image/x-png')}
    response = requests.post('http://wearmask.me/upload', files=data)
    file.close()
    os.remove(f'images/{name}.png')
    print(response.text)