from flask import Flask, request
app = Flask("region_detection")

from threading import Thread
import os
import numpy as np
import cv2
import utils.region_detect as region_detect_utils

import requests
CHAR_RECOGNITION_SERVICE_URL = os.environ.get('CHAR_RECOGNITION_SERVICE_URL') or 'http://localhost:5002'

cwd, _ = os.path.split(os.path.abspath(__file__))
# def prepare(yolo_anchors_path, yolo_model_path, east_model_path, east_json_path):
yolo_anchors_path = os.environ.get('yolo_anchor_path') or cwd + '/' + '../data/yolov3_anchors.txt'
yolo_model_path = os.environ.get('yolo_model_path') or cwd + '/' + '../data/yolov3.h5'

east_json_path = os.environ.get('east_json_path') or cwd + '/' + '../data/model.json'
east_model_path = os.environ.get('east_model_path') or cwd + '/' + '../data/east.h5'



region_detect_utils.prepare(yolo_anchors_path, yolo_model_path, east_json_path, east_model_path)

def detect_regions(frame_image, time_range):
    image = region_detect_utils.process_frame_for_yolo(frame_image)
    regions = region_detect_utils.detect_object(image)
    print('got {} regions'.format(len(regions)))

    # Send all regions to the char recognition service
    for region in regions[:2]:
        _, im_encoded = cv2.imencode('.jpg', np.array(region))
        res = requests.post(url=CHAR_RECOGNITION_SERVICE_URL,
                            data=im_encoded.tostring(),
                            headers={'x-time-range': time_range})
        print('region', res.status_code, res.text)

@app.route("/", methods=['POST'])
def handle_frame():
    r = request
    time_range = r.headers.get('x-time-range', type=str)

    if time_range == None:
        print('time-range not specified')
        return 'Missing time-range header', 400

    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    frame_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    region_detection_thread = Thread(target=detect_regions, args=[frame_img, time_range])
    region_detection_thread.start()

    return "frame recieved!"

app.run(port=5001)


