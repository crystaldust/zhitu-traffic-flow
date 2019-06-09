# model.keras_model.save_weights(model_savepath)
import tensorflow as tf
from keras import backend as K

def convert_model_to_pb(model, savedir, filename):
    frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, savedir, filename, as_text=False)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants

    graph = session.graph

    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()

        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
            frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)

            return frozen_graph


_anchor_cache = {}
anchors = []

def compute_backbone_shapes(config: object, image_shape: object) -> object:
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])

def get_anchors(image_shape, config):
    """Returns anchor pyramid for the given image size."""
    backbone_shapes = compute_backbone_shapes(config, image_shape)
    # Cache anchors and reuse if image shape is the same
    if not tuple(image_shape) in _anchor_cache:
        # Generate Anchors
        a = utils.generate_pyramid_anchors(
            config.RPN_ANCHOR_SCALES,
            config.RPN_ANCHOR_RATIOS,
            backbone_shapes,
            config.BACKBONE_STRIDES,
            config.RPN_ANCHOR_STRIDE)
        # Keep a copy of the latest anchors in pixel coordinates because
        # it's used in inspect_model notebooks.
        # TODO: Remove this after the notebook are refactored to not use it
        anchors = a
        # Normalize coordinates
        _anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
    return _anchor_cache[tuple(image_shape)]

def load_image(path):
    """Load the specified image and return a [H,W,3] Numpy array.
    """
    # Load image
    image = skimage.io.imread(path)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image

def mold_inputs(images, config):
    """Takes a list of images and modifies them to the format expected
    as an input to the neural network.
    images: List of image matrices [height,width,depth]. Images can have
        different sizes.

    Returns 3 Numpy matrices:
    molded_images: [N, h, w, 3]. Images resized and normalized.
    image_metas: [N, length of meta data]. Details about each image.
    windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
        original image (padding excluded).
    """
    molded_images = []
    image_metas = []
    windows = []
    for image in images:
        # Resize image
        # TODO: move resizing to mold_image()
        molded_image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)
        molded_image = mold_image(molded_image, config)
        # Build image_meta
        image_meta = compose_image_meta(
            0, image.shape, molded_image.shape, window, scale,
            np.zeros([config.NUM_CLASSES], dtype=np.int32))
        # Append
        molded_images.append(molded_image)
        windows.append(window)
        image_metas.append(image_meta)
    # Pack into arrays
    molded_images = np.stack(molded_images)
    image_metas = np.stack(image_metas)
    windows = np.stack(windows)
    return molded_images, image_metas, windows

def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta

from tensorflow.python.platform import gfile

def load_tf_model(sess, modelpath):
    with gfile.FastGFile(modelpath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()