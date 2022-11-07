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
from __future__ import division
import sys
import os
import argparse
import json
import cv2
import numpy as np
import sophon.sail as sail
import time
import datetime
import preprocessBMCV
import tools
def preprocess_handle(img, bmcv):
    
  preprocess_list = preprocessBMCV.get_midas_preprocess(bmcv)
  input_batch = preprocessBMCV.run_preprocess({"image":img},preprocess_list)
  input_batch = input_batch['image']
  return input_batch

def write_pfm(path, image, scale=1):
    """Write pfm file.

    Args:
        path (str): pathto file
        image (array): data
        scale (int, optional): Scale. Defaults to 1.
    """

    with open(path, "wb") as file:
        color = None

        if image.dtype.name != "float32":
            raise Exception("Image dtype must be float32.")

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif (
            len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
        ):  # greyscale
            color = False
        else:
            raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

        file.write("PF\n" if color else "Pf\n".encode())
        file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale

        file.write("%f\n".encode() % scale)

        image.tofile(file)

    
def write_depth(path, depth, bits=1):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    write_pfm(path + ".pfm", depth.astype(np.float32))

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))

    return

def inference(net, input_path, loops, tpu_id):
  """ Load a bmodel and do inference.
  Args:
   net: Loaded net
   input_path: Path to input file
   loops: Number of loops to run
   tpu_id: ID of TPU to use

  Returns:
    True for success and False for failure
  """
  # set configurations

  # get model info
  net_opt  = tools.get_prepare(net)
  print(net_opt)
  input_tensor, output_tensor = tools.build_inout_tensor(net_opt)
  net.set_io_mode(net_opt['graph_name'], sail.IOMode.SYSO)
  bmcv = tools.BMCV(net_opt['handle'])
  
  # pipeline of inference
  for i in range(loops):
    # read an image
    img = bmcv.imread(input_path)
    t1=time.time()
    data = preprocess_handle(img, bmcv)
    bmcv.totensor_(data, input_tensor)
    print(input_tensor, input_tensor.own_dev_data(), input_tensor.shape())
    t2=time.time()
    print("preprocess cost : %.3f second" % (t2 - t1))
    net.process(net_opt['graph_name'], {net_opt['input_name']:input_tensor},{net_opt['input_name']: net_opt['input_shape']} , {net_opt['output_name']:output_tensor})
    t1=time.time()
    print("infer cost : %.3f second" % (t1 - t2))
    output = output_tensor.asnumpy(net_opt['output_shape'])[0]
    write_depth("output", output, bits=2)
    t2=time.time()
    gray2colorSave(output)


def gray2colorSave(output, cmap=None):
    "cmap: cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_BONE, cv2.COLORMAP_RAINBOW"
    # output normalize to gray image 
    gray = output / output.max() * 255
    gray = gray.astype(np.uint8)
    # inferno cv2.COLORMAP_INFERNO
    color = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    cv2.imwrite("output-with-color.jpg", color)

if __name__ == '__main__':
  """ A Midas example.
  """
  PARSER = argparse.ArgumentParser(description='for sail midas_opencv_inference.py test')
  PARSER.add_argument('--bmodel', default='../dataset/midas_s_fp32b1.bmodel')
  PARSER.add_argument('--tpu_id', default=0, type=int, required=False)
  PARSER.add_argument('--input', default='../dataset/dog.jpg')
  ARGS = PARSER.parse_args()
  if not os.path.isfile(ARGS.input):
    print("Error: {} not exists!".format(ARGS.input))
    sys.exit(-2)

  preds = []
  processed=0

  # load bmodel to create the net
  net = sail.Engine(ARGS.bmodel, ARGS.tpu_id, sail.IOMode.SYSIO)

  inference(net, ARGS.input, 1, ARGS.tpu_id)


