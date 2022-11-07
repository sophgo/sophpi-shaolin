""" Copyright 2016-2022 by SOPHGO

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import sophon.sail as sail
import os 

def check_BMImage(obj):
    if isinstance(obj, sail.BMImage):
        return True 
    return False

# get the env variable 
def get_env_var(var_name, default_value=5):
    if var_name in os.environ:
        return os.environ[var_name]
    else:
        return default_value

def print_with_level(msg, level=3):
    if level == 3:
        print("[Warning] ", end="\t")
        print(msg)
    elif level >= 2:
        print("[Info] ", end="\t")
        print(msg)
    elif level >= 1:
        print("[Debug] ", end="\t")
        print(msg)
    else:
        print("[Error] ", end="\t")
        print(msg)

def img2bgr(img, bmcv):
    """ convert img yuv to bgr
    Only support yuv420p convert 2 bgr, other format will directly return img with warning
    Args:
        img (_type_): _description_
        bmcv (_type_): _description_
    """
    assert check_BMImage(img)
    if "YUV" in str(img.format()):
        img = bmcv.yuv2bgr(img)
    elif 'BGR' in str(img.format()):
        pass
    else:
        print_with_level("current img format is {} which is not supported".format(img.format()), 3)
    return img

class Message:
    
    @classmethod 
    def info(cls, msg):
        print_with_level(msg, 2)
        
    @classmethod
    def warning(cls, msg):
        print_with_level(msg, 3)
        
    @classmethod
    def error(cls, msg):
        print_with_level(msg, 0)
    
    @classmethod
    def debug(cls, msg):
        print_with_level(msg, 1)

def generate_alpha(mean, std=[1,1,1]):
    # (a0, b0), (a1, b1), (a2, b2) 
    if 0 in std:
        raise ValueError('std cannot be zero')
    alist = [1/i for i in std]
    blist = [-i*j for i, j in zip(mean, alist)]
    return (alist[0], blist[0]), (alist[1], blist[1]), (alist[2], blist[2])

class BMCV:
    def __init__(self, handle):
        
        self.handle = handle
        if self.handle is None:
            ValueError("Warning: No `handle` found, `imread` will not work")
                
        self.bmcv = sail.Bmcv(handle) 
        self.fns = {}
        
    def __str__(self):
        return "This is a BMCV interface"    
    
    def imread(self, path, convert2bgr=True, dtype = sail.ImgDtype.DATA_TYPE_EXT_FLOAT32):
        """ Read image from path
        if dtype is None, raise error
        Return a bmimage object 
        """
        if dtype is None:
            Message.error("dtype cannot be None")
            ValueError("dtype cannot be None")
        
        img = sail.BMImage()
        
        decoder = sail.Decoder(path)
        decoder.read(self.handle, img)
        
        if convert2bgr:
            img = img2bgr(img, self.bmcv)
        return img
    
    def resize(self,img, size, interpolation=0):
        if interpolation != 0:
            Message.warning("Only support interpolation nearest")
        res = self.bmcv.vpp_resize(img, size[0], size[1])
        return res 
    
    def tonumpy(self,img):
        assert check_BMImage(img)
        tensor = self.bmcv.bm_image_to_tensor(img)
        npdata = tensor.asnumpy()
        return npdata
    
    def totensor(self, img):
        tensor = self.bmcv.bm_image_to_tensor(img)
        return tensor 
    
    def totensor_(self,img, tensor):
        self.bmcv.bm_image_to_tensor(img, tensor)
    
    def resize_(self, img, img2, size, interpolation=0):
        # if interpolation != 0:
        #     raise ValueError("Only support interpolation nearest")
        self.bmcv.vpp_resize(img, img2, size[0], size[1])
        
    def imwrite(self, path, img):
        self.bmcv.imwrite(path, img)

    def batchImRead(self, img_paths, batch_size=1):
        """ Batch data
        """
        if batch_size > 1:
            
            imgs_0 = sail.BMImageArray4D()
            for idx, each_path in enumerate(img_paths):
                decoder = sail.Decoder(each_path)
                decoder.read_(self.handle, imgs_0)
            return imgs_0
        else:
            return self.imread(img_paths[0])

    def diffwithnp(self, img, npdata):
        assert check_BMImage(img)
        tensor = self.bmcv.bm_image_to_tensor(img)
        npdata2 = tensor.asnumpy()
        return npdata2 - npdata
    
    def normalize(self, img, mean=[0,0,0], std=[1,1,1]):
        # attention  img dtype should be float32 otherwise create another img 
        alpha = generate_alpha(mean, std)
        if img.dtype() != sail.ImgDtype.DATA_TYPE_EXT_FLOAT32:
            Message.info("img dtype should be float32. Current dtype is {}".format(img.dtype()))
            img0 = sail.BMImage(self.handle, img.width(), img.height(), img.format(), sail.ImgDtype.DATA_TYPE_EXT_FLOAT32)
            self.bmcv.convert_to(img, img0, alpha)
            img = img0
        else:
            img = self.bmcv.convert_to(img, alpha)
        return img 
    
    def normalize_(self, img, target, mean=[0,0,0], std=[1,1,1]):
        alpha = generate_alpha(mean, std)
        self.bmcv.convert_to(img, target, alpha)
    
    def np2bmimage(self, npdata):
        tensor = self.bmcv.Tensor(npdata)
        img = self.bmcv.tensor_to_bm_image(tensor)
        return img
    
    # def __getattribute__(self, __name: str):
    #     if __name in self.fns:
    #         return self.fns[__name]
        
    #     return getattr(self.bmcv, __name)
    
    # def registerfn(self, fn, fname=None):
        
    #     fname = fn.__name__ if fname is None else fname
    #     self.fns[fname] = fn


def get_prepare(net):
    """ Get input and output name and type
    Return a dict with 
        graph_name, input_name, input_dtype, output_name, output_dtype, input_shape, output_shape
    """
    handle = net.get_handle()
    graph_name = net.get_graph_names()[0]
    input_name = net.get_input_names(graph_name)[0]
    output_name = net.get_output_names(graph_name)[0]
    input_dtype= net.get_input_dtype(graph_name, input_name)
    output_dtype = net.get_output_dtype(graph_name, output_name) 
    input_shape = net.get_input_shape(graph_name, input_name) # get the maxmium dimension of the tensor 
    output_shape= net.get_output_shape(graph_name, output_name)
    res= { "graph_name": graph_name, "input_name": input_name, 
          "input_dtype": input_dtype, "output_name": output_name, 
          "output_dtype": output_dtype, "input_shape": input_shape, 
          "output_shape": output_shape, "handle": handle}
    return res

def build_inout_tensor(opt: dict):
    """ build sail tensor with opt 
    opt need contains 'input_dtype' 'input_shape', 'handle', 'output_dtype', 'output_shape'
    ValueError will be raised if opt is not valid 
    """
    if not isinstance(opt, dict):
        raise ValueError("opt must be a dict")
    if "input_dtype" not in opt or "input_shape" not in opt or "handle" not in opt:
        raise ValueError("opt must contains 'input_dtype' 'input_shape' 'handle'")
    if "output_dtype" not in opt or "output_shape" not in opt:
        raise ValueError("opt must contains 'output_dtype' 'output_shape'")
    dtype = opt["input_dtype"]
    shape = opt["input_shape"]
    handle = opt["handle"]
    input_tensor = sail.Tensor(handle, shape, dtype, False, False)
    output_tensor= sail.Tensor(handle, shape, dtype, True, True)
    
    return input_tensor, output_tensor

