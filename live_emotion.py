import cv2
import sys
#sys.path.insert(0,'../')

from src.emotion.utils import load_model as load_emotion
from src.detect_face.inference import load_detection_model as load_face
from src.utils import *

vid = cv2.VideoCapture(0)
face_detector = load_face()
emotion_detector = load_emotion()


while True:
    _,frame = vid.read()
    image = bytes2array(request.data)
    face,bbox = face_detector.find_biggest_face(image)
    emotion = emotion_detector.predict_array(face)['emotion']
    draw_bbox(image,bbox)
    cv2.imshow(emotion,face)
    if 0x00==orq('q') & cv2.waitkey(1):
        break



