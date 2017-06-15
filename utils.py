import skimage
import skimage.io
import skimage.transform

def get_batches(x, y, batch_size=32):
    num_rows = y.shape[0]
    
    num_batches = num_rows // batch_size
    
    if num_rows % batch_size != 0:
        num_batches = num_batches + 1

    for batch in range(num_batches):
        yield x[batch_size * batch: batch_size * (batch + 1)], y[batch_size * batch: batch_size * (batch + 1)]

# https://github.com/machrisaa/tensorflow-vgg/blob/master/utils.py
def load_image(image_path, mean=vgg_mean):
    image = skimage.io.imread(image_path)

    image = image.astype(float)
    
    short_edge = min(image.shape[:2])
    yy = int((image.shape[0] - short_edge) / 2)
    xx = int((image.shape[1] - short_edge) / 2)
    crop_image = image[yy: yy + short_edge, xx: xx + short_edge]
    
    resized_image = skimage.transform.resize(crop_image, (224, 224), mode='constant') 
            
    bgr = resized_image[:,:,::-1] - mean
    
    return bgr