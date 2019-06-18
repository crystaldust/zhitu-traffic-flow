from flask import Flask, request
app = Flask("ocr_service")

import numpy as np
import cv2
import os
import utils.ocr as ocr_utils
crnn_model_path = os.environ.get('crnn_model_path') or os.getcwd() + '/../../data/crnn.h5'
ocr_utils.prepare_model(crnn_model_path)

def store_text(text):
    print('Assume we have a storage service for car brand, making sure they are not duplicated in the same time range')

@app.route("/", methods=['POST'])
def ocr():
    r = request
    time_range = r.headers.get('x-time-range', type=str)

    if time_range == None:
        print('time-range not specified')
        return 'Missing time-range header', 400
    try:
        image_ary = np.frombuffer(r.data, np.uint8)
        img = cv2.imdecode(image_ary, cv2.IMREAD_COLOR)
    except Exception as e:
        print(e)
        return "Invalid image data", 400

    try:
        txt = ocr_utils.detect_text(img)
        store_text(txt)
        return txt, 200, {'Content-Type': 'text/plain'}
    except Exception as e:
        print('failed to detect text', e)
        return "text detection failed", 500

app.run(port=5002)
