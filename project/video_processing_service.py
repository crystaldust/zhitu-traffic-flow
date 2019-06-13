from flask import Flask, request
app = Flask("video_processing")

from threading import Thread
import os
import cv2
import utils.video_processing as video_utils

cwd, _ = os.path.split(os.path.abspath(__file__))
REGION_DETECTION_SERVICE_URL = os.environ.get('OBJECT_DETECTION_SERVICE_URL') or 'http://localhost:5001'

import requests


def process_video_frames(filepath, time_range):
    frames = video_utils.extract_frames(filepath, gap=5)
    ## Send all frames to the region detecting service
    for i in range(5):
    # for i in range(len(frames)):
        frame = frames[i]
        _, im_encoded = cv2.imencode('.jpg', frame)
        res = requests.post(url=REGION_DETECTION_SERVICE_URL,
                            data=im_encoded.tostring(),
                            headers={'x-time-range': time_range})
        print('video{}, frame@{}, stauts:{}'.format(time_range, i, res.status_code))

@app.route("/", methods=['POST'])
def handle_video():
    r = request
    time_range = r.headers.get('x-time-range', type=str)

    if time_range == None:
        print('time-range not specified')
        return 'Missing time-range header', 400
    video_savepath = 'video_{}.mp4'.format(time_range)
    with open(video_savepath, 'wb') as f:
        f.write(r.data)

    abs_video_savepath = cwd + '/' + video_savepath
    # Handle the video frames
    video_analysis_thread = Thread(target=process_video_frames, args=[abs_video_savepath, time_range])
    video_analysis_thread.start()

    return "video recieved!"

app.run(port=5000)


