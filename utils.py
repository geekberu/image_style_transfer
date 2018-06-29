import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

slim = tf.contrib.slim

# IMAGE_SIZE = 512
TV_WEIGHT = 1e2


def check_content_img(img_path):
    img = cv2.imread(img_path)
    shape = img.shape
    if shape[0] > 600:
        return False
    elif shape[1] > 600:
        return False
    return True


def load_content_img(img_path):
    img = cv2.imread(img_path)
    shape = img.shape
    img = np.array(img, dtype='float32')
    images = tf.expand_dims(img, 0)
    # images = tf.concat([images, images, images, images], 0)  # batch is 4
    # cv2.imwrite('./content.jpg', img)

    return images, shape


def load_style_img(img_path, resize_shape):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (resize_shape[1], resize_shape[0]))
    img = np.array(img, dtype='float32')
    images = tf.expand_dims(img, 0)
    # images = tf.concat([images, images, images, images], 0)  # batch is 4
    # cv2.imwrite('./style.jpg', img)

    return images


def cal_content_loss(f):
    gen_f, img_f, _ = tf.split(f, 3, 0)
    content_loss = tf.nn.l2_loss(gen_f - img_f) / tf.to_float(tf.size(gen_f))
    return content_loss


def cal_style_loss(f1, f2, f3, f4):
    gen_f, _, style_f = tf.split(f1, 3, 0)
    size = tf.size(gen_f)
    style_loss = tf.nn.l2_loss(gram(gen_f) - gram(style_f)) * 2 / tf.to_float(size)

    gen_f, _, style_f = tf.split(f2, 3, 0)
    size = tf.size(gen_f)
    style_loss += tf.nn.l2_loss(gram(gen_f) - gram(style_f)) * 2 / tf.to_float(size)

    gen_f, _, style_f = tf.split(f3, 3, 0)
    size = tf.size(gen_f)
    style_loss += tf.nn.l2_loss(gram(gen_f) - gram(style_f)) * 2 / tf.to_float(size)

    gen_f, _, style_f = tf.split(f4, 3, 0)
    size = tf.size(gen_f)
    style_loss += tf.nn.l2_loss(gram(gen_f) - gram(style_f)) * 2 / tf.to_float(size)

    return style_loss


def cal_tv_loss(gen_img, content_shape):
    # total variation denoising
    tv_y_size = tensor_size(gen_img[:, 1:, :, :])
    tv_x_size = tensor_size(gen_img[:, :, 1:, :])
    tv_loss = TV_WEIGHT * 2 * (
            (tf.nn.l2_loss(gen_img[:, 1:, :, :] - gen_img[:, :content_shape[1] - 1, :, :]) /
             tv_y_size) +
            (tf.nn.l2_loss(gen_img[:, :, 1:, :] - gen_img[:, :, :content_shape[2] - 1, :]) /
             tv_x_size))
    return tv_loss


def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

    return grams


def tensor_size(tensor):
    from operator import mul
    from functools import reduce
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb


def image_process(gen_img, content_img_path):
    # Luminosity transfer steps:
    # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
    # 2. Convert stylized grayscale into YUV (YCbCr)
    # 3. Convert original image into YUV (YCbCr)
    # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
    # 5. Convert recombined image from YUV back to RGB

    content_img = cv2.imread(content_img_path)
    img = cv2.resize(content_img, (IMAGE_SIZE, IMAGE_SIZE))
    original_image = np.clip(img, 0, 255)
    styled_image = np.clip(gen_img, 0, 255)
    # 1
    styled_grayscale = rgb2gray(styled_image)
    styled_grayscale_rgb = gray2rgb(styled_grayscale)

    # 2
    styled_grayscale_yuv = np.array(Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr'))

    # 3
    original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr'))

    # 4
    w, h, _ = original_image.shape
    combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
    combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
    combined_yuv[..., 1] = original_yuv[..., 1]
    combined_yuv[..., 2] = original_yuv[..., 2]

    # 5
    img_out = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))

    return img_out
