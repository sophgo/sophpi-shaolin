
import numpy as np
import cv2
import math


class ResizeBMCV(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
        bmcv = None,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height
        self.bmcv = bmcv
        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        w = sample["image"].width()
        h = sample["image"].height()
        width, height = self.get_size(
            w, h
        )

        # resize sample
        sample["image"] = self.bmcv.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        return sample


class NormalizeImage(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std, bmcv=None):
        self.__mean = mean
        self.__std = std
        self.bmcv = bmcv

    def __call__(self, sample):
        sample["image"] = self.bmcv.normalize(sample["image"], self.__mean, self.__std)

        return sample


class Scale(object):
    def __init__(self, bmcv=None):
        self.bmcv = bmcv
        self.std = [255.0, 255.0, 255.0]
        self.mean = [0,0,0]
        
    def __call__(self, sample:dict):
        sample['image'] = self.bmcv.normalize(sample['image'], self.mean, self.std)
        return sample
        

def get_midas_preprocess(bmcv):
    """get midas model preprocess (only support model type == MiDaS_small)
    refer to https://pytorch.org/hub/intelisl_midas_v2/
    Returns:
        List: A list of preprocess function
    """
    
    net_w = 256
    net_h = 256
    resize_mode = "upper_bound"
    hub = Scale(bmcv=bmcv)
    resize = ResizeBMCV(
        net_w, 
        net_h, 
        resize_target=None,
        keep_aspect_ratio=True,
        ensure_multiple_of=32,
        resize_method=resize_mode,
        image_interpolation_method=cv2.INTER_CUBIC,
        bmcv = bmcv
        )    
    normalization = NormalizeImage(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], bmcv = bmcv
    )
    total_preprocess = [
        resize,
        hub,
        normalization
    ] # for bmcv model we need resize first and then scale to [0,1] range 
    return total_preprocess

def run_preprocess(data, preprocess_list):
    """general preprocess tools (simulate to torchvision compose). 'preprocess_list' is a list of functions to handle data and return new data. 

    Args:
        data (dict|ndarray): must match with preprocess_list function argments
        preprocess_list (list(function)): _description_

    Returns:
        data
    """
    for fn in preprocess_list:
        data = fn(data)
        print(data['image'].width(), data['image'].height())
    return data