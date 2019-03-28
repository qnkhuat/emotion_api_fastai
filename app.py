from flask import Flask, jsonify, make_response, request, abort, redirect
import logging
import base64
import json
from src.emotion.utils import prepare_input, load_model as load_emotion
from src.utils import bytes2array
from src.detect_face.inference import load_detection_model as load_face
import time

app = Flask(__name__)
emotion_detector = load_emotion()
face_detector = load_face()

@app.route('/emotion/analyze', methods=['POST'])
def analyze():
    try:
        image = bytes2array(request.data)
        face = face_detector.find_biggest_face(image)
        result = emotion_detector.predict_array(face)
        return make_response(jsonify(result), 200)
    except Exception as err:
        logging.error('An error has occurred whilst processing the file: "{0}"'.format(err))
        abort(400)

@app.errorhandler(400)
def bad_request(error):
    print(error)
    return make_response(jsonify({'message': 'We cannot process the file sent in the request.'}), 400)

@app.errorhandler(404)
def not_found(error):
    print(error)
    return make_response(jsonify({'message': 'Resource not found.'}), 404)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8085)
