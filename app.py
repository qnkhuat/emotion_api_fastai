from flask import Flask, jsonify, make_response, request, abort, redirect
import logging
import base64
import json
from src.emotion.utils import prepare_input, load_model as load_emotion
from src.utils import encode_payload
from src.detect_face.inference import load_detection_model as load_face

app = Flask(__name__)
emotion_detector = load_emotion()
face_detector = load_face()

@app.route('/emotion/analyze', methods=['POST'])
def upload():
    data = encode_payload(request.data)
    try:
        image_base64 = data['image']
        image = base64_2_array(image_base64)
        face = face_detector.find_biggest_face(image)
        pred = emotion_detector.predict_array(face)
        emotion = str(pred[0])
        result = {'emotion':emotion}
        return make_response(jsonify(result), 200)
    except Exception as err:
        logging.error('An error has occurred whilst processing the file: "{0}"'.format(err))
        abort(400)

@app.errorhandler(400)
def bad_request(erro):
    return make_response(jsonify({'error': 'We cannot process the file sent in the request.'}), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Resource no found.'}), 404)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8085)
