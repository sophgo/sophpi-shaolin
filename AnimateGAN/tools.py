import os,cv2
import numpy as np


def load_test_data(image_path, size):
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    source_size = img.shape[:2]
    img = preprocessing(img,size)
    img = np.expand_dims(img, axis=0)
    return img, source_size

def load_test_data_from_pil(image, size):
    img = image
    source_size = img.shape[:2]
    img = preprocessing(img,size)
    img = np.expand_dims(img, axis=0)
    return img, source_size

def preprocessing(img, size):
    w,h = size
    # h, w = img.shape[:2]
    # if h <= size[0]:
    #     h = size[0]
    # else:
    #     x = h % 32
    #     h = h - x

    # if w < size[1]:
    #     w = size[1]
    # else:
    #     y = w % 32
    #     w = w - y
    # the cv2 resize func : dsize format is (W ,H)
    img = cv2.resize(img, (w, h))
    return img/127.5 - 1.0

def inverse_transform(images):
    images = (images + 1.) / 2 * 255
    # The calculation of floating-point numbers is inaccurate, 
    # and the range of pixel values must be limited to the boundary, 
    # otherwise, image distortion or artifacts will appear during display.
    images = np.clip(images, 0, 255)
    return images.astype(np.uint8)

def read_img(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assert len(img.shape)==3
    return img

# Calculates the average brightness in the specified irregular image
def calculate_average_brightness(img):
    # Average value of three color channels
    B = img[..., 0].mean()
    G = img[..., 1].mean()
    R = img[..., 2].mean()

    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    return brightness, B, G, R

# Adjusting the average brightness of the target image to the average brightness of the source image
def adjust_brightness_from_src_to_dst(dst, src,path=None,if_show=None, if_info=None):
    brightness1, B1, G1, R1 = calculate_average_brightness(src)
    brightness2, B2, G2, R2 = calculate_average_brightness(dst)
    brightness_difference = brightness1 / brightness2

    if if_info:
        print('Average brightness of original image', brightness1)
        print('Average brightness of target', brightness2)
        print('Brightness Difference between Original Image and Target', brightness_difference)

    # According to the average display brightness
    dstf = dst * brightness_difference

    # According to the average value of the three-color channel
    # dstf = dst.copy().astype(np.float32)
    # dstf[..., 0] = dst[..., 0] * (B1 / B2)
    # dstf[..., 1] = dst[..., 1] * (G1 / G2)
    # dstf[..., 2] = dst[..., 2] * (R1 / R2)

    # To limit the results and prevent crossing the border,
    # it must be converted to uint8, otherwise the default result is float32, and errors will occur.
    dstf = np.clip(dstf, 0, 255)
    dstf = np.uint8(dstf)

    ma,na,_ = src.shape
    mb,nb,_ = dst.shape
    result_show_img = np.zeros((max(ma, mb), 3 * max(na, nb), 3))
    result_show_img[:mb, :nb, :] = dst
    result_show_img[:ma, nb:nb + na, :] = src
    result_show_img[:mb, nb + na:nb + na + nb, :] = dstf
    result_show_img = result_show_img.astype(np.uint8)

    if if_show:
        cv2.imshow('-', cv2.cvtColor(result_show_img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if path != None:
        cv2.imwrite(path, cv2.cvtColor(result_show_img, cv2.COLOR_BGR2RGB))

    return dstf

# Adjusting the average brightness of the target image to the average brightness of the source image
def adjust_brightness_from_src_to_dst_web(dst, src,):
    brightness1, B1, G1, R1 = calculate_average_brightness(src)
    brightness2, B2, G2, R2 = calculate_average_brightness(dst)
    brightness_difference = brightness1 / brightness2
    # According to the average display brightness
    dstf = dst * brightness_difference

    # To limit the results and prevent crossing the border,
    # it must be converted to uint8, otherwise the default result is float32, and errors will occur.
    dstf = np.clip(dstf, 0, 255)
    dstf = np.uint8(dstf)

    ma,na,_ = src.shape
    mb,nb,_ = dst.shape
    result_show_img = np.zeros((max(ma, mb), 3 * max(na, nb), 3))
    result_show_img[:mb, :nb, :] = dst
    result_show_img[:ma, nb:nb + na, :] = src
    result_show_img[:mb, nb + na:nb + na + nb, :] = dstf
    result_show_img = result_show_img.astype(np.uint8)

    return dstf


def get_save_images(images, size=None, augment = False, source=None):
    fake = inverse_transform(images.squeeze())
    
    if size is not None:
        fake = cv2.resize(fake, size)
    
    if augment:
        fake = adjust_brightness_from_src_to_dst_web(fake, source)

    return fake

def save_images(images, image_path, size=None, photo_path = None):
    fake = inverse_transform(images.squeeze())
    
    if size is not None:
        fake = cv2.resize(fake, size)
    
    if photo_path:
        return imsave(adjust_brightness_from_src_to_dst(fake, read_img(photo_path)),  image_path)
    else:
        return imsave(fake, image_path)
    
def imsave(images, path):
    return cv2.imwrite(path, cv2.cvtColor(images, cv2.COLOR_BGR2RGB))

