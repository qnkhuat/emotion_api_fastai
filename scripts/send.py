# https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594
import requests
import cv2
import base64
import json

url = 'http://0.0.0.0:8085/emotion/analyze'
img_file = '/home/qnkhuat/AI/vision/emotion_api/scripts/images/test.png'

with open(img_file, "rb") as image_file:
    payload= base64.b64encode(image_file.read())
    #payload = image_file.read().encode('base64')

#url = 'http://localhost:5000'
## prepare headers for http request
#content_type = 'image/jpeg'
headers = {'content-type' : 'application/json'}
headers = {'content-type' : 'base64'}
#img = cv2.imread('lena.jpg')
## encode image as jpeg
#_, img_encoded = cv2.imencode('.jpg', img)
#payload = {'image':encoded_string}
#payload = json.dumps(payload)
# send http request with image and receive response
response = requests.post(url, data=payload, headers=headers)
#response = requests.post(url, data=payload)
# decode response
print(response.text)
