import numpy as np
import tensorflow as tf

## 加载CRNN模型
from src.ocr.crnn.model import get_model, decode
# K.clear_session()
characters = u"""京津沪渝冀晋辽吉黑苏浙皖闽赣鲁豫鄂湘粤琼川黔贵滇云陕秦甘陇青藏桂蒙宁新澳台ABCDEFGHIJKLMNOPQRSTUVWXYZ·0123456789 """
nclass=len(characters)
height = 32

# Allow GPU memory growth
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

graph = None
crnn_model = None
def prepare_model(crnn_model_path):
    global graph
    global crnn_model

    if graph == None or crnn_model == None:
        _, crnn_model = get_model(height, nclass, learning_rate=0.0001)
        crnn_model.load_weights(crnn_model_path)
        graph = tf.get_default_graph()

from PIL import Image
def detect_text(image_array):
    image = Image.fromarray(image_array)

    scale = image.size[1] * 1.0 / 32
    print(scale)
    w = int(image.size[0] / scale)
    print(w)
    image = image.resize((w, 32))
    image = image.convert('L')

    image = np.array(image).astype(np.float32) / 255.0

    X = image.reshape((32, w, 1))
    X = np.array([X])
    # 预测
    global graph
    with graph.as_default():
        y_pred = crnn_model.predict(X)
    texts = decode(y_pred)
    return texts

def decode(pred):
    t = pred.argmax(axis=2)[0]
    length = len(t)
    char_list = []
    n = len(characters)
    for i in range(length):
        if t[i] != n and (not (i > 0 and t[i - 1] == t[i])):
            char_list.append(characters[t[i]])
            print(char_list)
    return u''.join(char_list)

