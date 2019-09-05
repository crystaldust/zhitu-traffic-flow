import cv2
import os
import numpy as np
import tensorflow as tf
import keras.backend as K

## 加载CRNN模型
import string

from src.ocr.crnn.models import CRNN_STN

class CRNNConfig:
    def __init__(self):
        self.characters = '0123456789'+string.ascii_lowercase
        self.width = 200
        self.height = 31
        self.nb_channels = 1
        self.lstm_nb_units = [128, 128]
        self.conv_filter_size = [64, 128, 256, 256, 512, 512, 512]
        self.dropout_rate = 0.25
        self.label_len = 16

cfg = CRNNConfig()

# Allow GPU memory growth
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# The ref to tf default graph is essential
graph = None
crnn_model = None
def prepare_model(crnn_model_path):
    global crnn_model
    global graph

    _, crnn_model = CRNN_STN(cfg)
    print('debug: load from', crnn_model_path)
    crnn_model.load_weights(crnn_model_path)

    graph = tf.get_default_graph()

def pad_image(img, img_size, nb_channels):
    # img_size : (width, height)
    # loaded_img_shape : (height, width)
    img_reshape = cv2.resize(img, (int(img_size[1] / img.shape[0] * img.shape[1]), img_size[1]))
    if nb_channels == 1:
        padding = np.zeros((img_size[1], img_size[0] - int(img_size[1] / img.shape[0] * img.shape[1])), dtype=np.int32)
    else:
        padding = np.zeros((img_size[1], img_size[0] - int(img_size[1] / img.shape[0] * img.shape[1]), nb_channels), dtype=np.int32)
    img = np.concatenate([img_reshape, padding], axis=1)
    return img

def resize_image(img, img_size):
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
    img = np.asarray(img)
    return img

def preprocess_image(img, cfg):
    if img.shape[1] / img.shape[0] < 6.4:
        img = pad_image(img, (cfg.width, cfg.height), cfg.nb_channels)
    else:
        img = resize_image(img, (cfg.width, cfg.height))
    if cfg.nb_channels == 1:
        img = img.transpose([1, 0])
    else:
        img = img.transpose([1, 0, 2])
    img = np.flip(img, 1)
    img = img / 255.0
    if cfg.nb_channels == 1:
        img = img[:, :, np.newaxis]
    return img

from src.ocr.crnn.utils import preprocess_image
def detect_text(src_img):
    img = preprocess_image(src_img, cfg)
    global graph
    with graph.as_default():
        y_pred = crnn_model.predict(img[np.newaxis, :, :, :])
        shape = y_pred[:, 2:, :].shape
        ctc_decode = K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0])*shape[1])[0][0]
        ctc_out = K.get_value(ctc_decode)[:, :cfg.label_len]
        result_str = ''.join([cfg.characters[c] for c in ctc_out[0]])
        result_str = result_str.replace('-', '')
        return result_str
