from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf



import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False


import numpy as np
import time
import functools

content_path = tf.keras.utils.get_file('belfry.jpg','https://storage.googleapis.com/khanhlvg-public.appspot.com/arbitrary-style-transfer/belfry-2611573_1280.jpg')
print(content_path)
style_path = tf.keras.utils.get_file('style23.jpg','https://storage.googleapis.com/khanhlvg-public.appspot.com/arbitrary-style-transfer/style23.jpg')

style_predict_path = tf.keras.utils.get_file('style_predict.tflite', 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/arbitrary_style_transfer/style_predict_quantized_256.tflite')
style_transform_path = tf.keras.utils.get_file('style_transform.tflite', 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/arbitrary_style_transfer/style_transfer_quantized_dynamic.tflite')

def load_img(path_to_img) :
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels = 3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

def preprocess_style_image(style_image) :
    target_dim = 256
    shape = tf.cast(tf.shape(style_image)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    style_shape = tf.image.resize(style_image, new_shape)

    style_image = tf.image.resize_with_crop_or_pad(style_image, target_dim, target_dim)
    return style_image
    
def preprocess_content_image(content_image) :
    shape = tf.shape(content_image)[1:-1]
    short_dim = min(shape)
    content_image = tf.image.resize_with_crop_or_pad(content_image, short_dim, short_dim)
    return content_image

content_image = load_img(content_path)
style_image = load_img(style_path)

preprocessed_content_image = preprocess_content_image(content_image)
preprocessed_style_image = preprocess_style_image(style_image)

print('STYLE IMAGE SHAPE:', preprocessed_style_image.shape)
print('CONTENT IMAGE SHAPE:', preprocessed_content_image.shape)

def imshow(image, title = None) :
    if len(image.shape) > 3 :
        image = tf.squeeze(image, axis = 0)
    plt.imshow(image)
    if title :
        plt.title(title)

plt.subplot(1, 2, 1)
imshow(preprocessed_content_image, 'CONTENT IMAGE')

plt.subplot(1, 2, 2)
imshow(preprocessed_style_image, 'STYLE IMAGE')

def run_style_predict(preprocessed_style_image) :
    interpreter = tf.lite.Interpreter(model_path = style_predict_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], preprocessed_style_image)
    interpreter.invoke()
    style_bottleneck = interpreter.tensor(interpreter.get_output_details()[0]['index'])()
    return style_bottleneck

style_bottleneck = run_style_predict(preprocessed_style_image)
print('STYLE BOTTLENECK SHAPE:', style_bottleneck.shape)

def run_style_transform(style_bottleneck, preprocessed_content_image) :
    interpreter = tf.lite.Interpreter(model_path = style_transform_path)

    input_details = interpreter.get_input_details()
    interpreter.resize_tensor_input(input_details[0]['index'],
                                    preprocessed_content_image.shape)
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_details[0]['index'], preprocessed_content_image)
    interpreter.set_tensor(input_details[1]['index'], style_bottleneck)
    interpreter.invoke()

    stylized_image = interpreter.tensor(
        interpreter.get_output_details()[0]['index']
    )()
    return stylized_image

stylized_image = run_style_transform(style_bottleneck, preprocessed_content_image)

imshow(stylized_image, 'STYLIZED IMAGE')

# blending

style_bottleneck_content = run_style_predict(
    preprocess_style_image(content_image)
)
content_blending_ratio = 0.5
style_bottleneck_blended = content_blending_ratio * style_bottleneck_content + (1 - content_blending_ratio) * style_bottleneck

stylized_image_blended = run_style_transform(style_bottleneck_blended,
                                             preprocessed_content_image)

imshow(stylized_image_blended, 'BLENDED STYLIZED IMAGE')
