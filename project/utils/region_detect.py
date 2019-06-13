from src.yolo3.model import yolo_eval
from src.yolo3.utils import letterbox_image

from PIL import Image
from keras.models import load_model, model_from_json
import keras.backend as K

# Allow GPU memory growth
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# sess = K.get_session()


import numpy as np
import tensorflow as tf

from src.ocr.east.eval import resize_image, detect, sort_poly

boxes = None
scores = None
classes = None
east_model, yolo_model = [None, None]
graph = None
from src.yolo3.model import yolo_body
from keras.layers import Input

input_image_shape = K.placeholder(shape=(2,))
iou = 0.45
score = 0.3

def prepare(yolo_anchors_path, yolo_model_path, east_json_path, east_model_path):
    # Init
    global east_model, yolo_model
    ## 准备YoloV3先验Anchor
    with open(yolo_anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    ## 加载YoloV3模型文件

    yolo_model = yolo_body(Input(shape=(None, None, 3)), len(anchors) // 3, 80)
    yolo_model.load_weights(yolo_model_path)


    global boxes
    global scores
    global classes

    boxes, scores, classes = yolo_eval(
        yolo_model.output,
        anchors,
        80,
        input_image_shape,
        score_threshold=score,
        iou_threshold=iou)

    ## 加载EAST模型
    with open(east_json_path, 'r') as f:
        loaded_model_json = f.read()
    # 读取模型结构
    east_model = model_from_json(loaded_model_json, custom_objects={'tf': tf, 'RESIZE_FACTOR': 2})
    # 读取权重文件
    east_model.load_weights(east_model_path)

    global graph
    # , sess
    graph = tf.get_default_graph()
    # sess = K.get_session()

def process_frame_for_yolo(frame):
    height = int(frame.shape[0] / 2)
    width = int(frame.shape[1] / 2)

    new_image_size = (width, height)
    # resized_frame = cv2.resize(frame, dsize=new_image_size)  # 无需差值
    image = Image.fromarray(frame[:, :, ::-1])

    return image

def detect_object(image):
    # resize input image
    new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
    boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')

    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)
    image_data.shape


    regions = []
    image_area = image.size[0] * image.size[1]
    REGION_THRESHOLD = 0.05

    # BDD100K 数据集中的分类信息
    # {"id": 1, "name": "person"},
    # {"id": 2, "name": "rider"},
    # {"id": 3, "name": "car"},
    # {"id": 4, "name": "bus"},
    # {"id": 5, "name": "truck"},
    # {"id": 6, "name": "bike"},
    # {"id": 7, "name": "motor"},
    # {"id": 8, "name": "traffic light"},
    # {"id": 9, "name": "traffic sign"},
    # {"id": 10, "name": "train"}

    vichel_class_ids = [3, 4, 5, 7]


    def is_big_enough(region, threshold):
        w, h = region.size
        return w * h / image_area >= threshold

        # K.clear_session()
        # tf.get_default_graph
    with graph.as_default():
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = c
        box = out_boxes[i]
        score = out_scores[i]

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        # print(predicted_class, score, (left, top), (right, bottom))
        #     region = image.crop((box[1], box[0], box[3], box[2]))
        region = image.crop((left, top, right, bottom))

        if is_big_enough(region, REGION_THRESHOLD):
            regions.append(region)
    return regions

def extract_text_areas(image):
    img = np.array(image)

    # 图片尺寸调整
    img_resized, (ratio_h, ratio_w) = resize_image(img)
    img_resized = (img_resized / 127.5) - 1
    global graph
    with graph.as_default():
        # 对图片进行文本区域预测，完成FCN部分
        score_map, geo_map = east_model.predict(img_resized[np.newaxis, :, :, :])
        # 对预测结果进行NMS计算，完成NMS部分
        boxes, timer = detect(score_map=score_map, geo_map=geo_map)

    text_areas = []
    if boxes is not None:
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h
    for box in boxes:
        # to avoid submitting errors
        box = sort_poly(box.astype(np.int32))
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue
        # cv2 BGR格式的图片，需要将最后一个维度(颜色通道)顺序逆转
        # cv2.polylines(img[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),thickness=2)

        # 我们已经统一使用RGB格式，这里无需反转img的颜色通道
        # cv2.polylines(img, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),thickness=2)

        left = np.min(box[:, 0])
        right = np.max(box[:, 0])

        top = np.min(box[:, 1])
        bottom = np.max(box[:, 1])
        text_area = img[top: bottom, left: right]
        text_areas.append(text_area)
    return text_areas
