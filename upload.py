import requests
import cv2
import os
import numpy as np

def upload(name,img):
    print("uploading!")
    cv2.imwrite(f'images/{name}.png',img)
    file = open(f'images/{name}.png','rb')
    data = {"file" : (str(name)+".png",file, 'image/x-png')}
    result = requests.post('http://wearmask.me/upload', files=data)
    file.close()
    os.remove(f'images/{name}.png')
    print(result.text.strip())
    pass

def upload_pic(path):
    file = open(path,'rb')
    data = {"file" : ("test10.png", file, 'image/x-png')}
    result = requests.post('http://wearmask.me/upload', files=data)
    file.close()
    pass

upload_pic("download.png")