import tensorflow as tf
import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]
RANDOM_RESIZE_METHOD = -1

def load_image(filename_queue, size=None, resize_method=tf.image.ResizeMethod.BILINEAR):
    image_file = tf.read_file(filename_queue)
    image = tf.image.decode_jpeg(image_file)
    
    image = tf.image.convert_image_dtype(image, tf.float32)

    if size != None:
        if resize_method==RANDOM_RESIZE_METHOD:
            #assume random resize method
            random_method = tf.random_uniform(dtype=tf.int32, minval=0, maxval=3, shape=[])
            # only random at construction time
            image = tf.image.resize_images(image, size, method=np.random.randint(0, 3))
        else:
            image = tf.image.resize_images(image, size, method=resize_method)
        
    return image

def vgg16_preprocess(image, shape, mean=VGG_MEAN):
    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.reverse(image, axis=[-1]) # RGB to BGR
    
    image = tf.cast(image, dtype=tf.float32)
    image = tf.subtract(image, mean)
    
    image.set_shape(shape)
    return image


def distort_image(image, random_rotate=True, random_flip=True):
    
    if random_flip:
        image = tf.image.random_flip_left_right(image)
    
    if np.random.choice([False, True]): # only random at construction time
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    
    if random_rotate:
        rot90 = tf.random_uniform(dtype=tf.int32, minval=1, maxval=3, shape=[])
        image = tf.image.rot90(image, k=rot90)
    
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image