import argparse
import cv2
import numpy as np
import sophon.sail as sail
import os
import time

# from utils.colors import _COLORS
# from utils.utils import *
from utils import *

opt = None



class YOLOV5(object):
    def __init__(self, bmodel_path, tpu_id, confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.1):
        # load bmodel
        self.net = sail.Engine(bmodel_path, tpu_id, sail.IOMode.SYSIO)
        self.handle = self.net.get_handle()

        # get model info
        self.graph_name = self.net.get_graph_names()[0]

        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_shape = self.net.get_input_shape(
            self.graph_name, self.input_name)
        self.input_w = int(self.input_shape[-1])
        self.input_h = int(self.input_shape[-2])
        self.input_shapes = {self.input_name: self.input_shape}
        self.input_dtype = self.net.get_input_dtype(
            self.graph_name, self.input_name)

        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)


        self.output_name = self.net.get_output_names(self.graph_name)[0]

        self.output_shape = self.net.get_output_shape(self.graph_name, self.output_name)
        self.output_dtype = self.net.get_output_dtype(self.graph_name, self.output_name)
        self.output_scale  = self.net.get_output_scale(self.graph_name, self.output_name)

        self.output = sail.Tensor(self.handle, self.output_shape, self.output_dtype, True, True)

        self.output_tensors = {self.output_name: self.output}

        is_fp32 = (self.input_dtype == sail.Dtype.BM_FLOAT32)
        # get handle to create input and output tensors
        self.bmcv = sail.Bmcv(self.handle)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)

        # bgr normalization
        self.ab = [x * self.input_scale for x in [0.003921568627451, 0, 0.003921568627451, 0, 0.003921568627451, 0]]

        # generate anchor
        self.nl = 3
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62,
                                              45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.anchor_grid = np.asarray(
            anchors, dtype=np.float32).reshape(self.nl, 1, -1, 1, 1, 2)
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])

        # post-process threshold
        scalethreshold = 1.0 if is_fp32 else 0.9
        self.confThreshold = confThreshold * scalethreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold * scalethreshold
        self.ration = 1


        print("input img_dtype:{}, input scale: {}, output scale: {} ".format(
            self.img_dtype, self.input_scale, self.output_scale))

    def compute_IOU(self, rec1, rec2):
        """
        :param rec1: (x0,y0,x1,y1)      
        :param rec2: (x0,y0,x1,y1)
        :return: 交并比IOU.
        """
        left_column_max = max(rec1[0], rec2[0])
        right_column_min = min(rec1[2], rec2[2])
        up_row_max = max(rec1[1], rec2[1])
        down_row_min = min(rec1[3], rec2[3])
        # 两矩形无相交区域的情况
        if left_column_max >= right_column_min or down_row_min <= up_row_max:
            return 0
        # 两矩形有相交区域的情况
        else:
            S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
            S_cross = (down_row_min - up_row_max) * \
                (right_column_min - left_column_max)
            return S_cross / S1

    def sigmoid(self, inx):  # 防止exp(-x)溢出
        indices_pos = np.nonzero(inx >= 0)
        indices_neg = np.nonzero(inx < 0)

        y = np.zeros_like(inx)
        y[indices_pos] = 1 / (1 + np.exp(-inx[indices_pos]))
        y[indices_neg] = np.exp(inx[indices_neg]) / \
            (1 + np.exp(inx[indices_neg]))

        return y

    def preprocess(self, img, c, h, w):

        img_h, img_w, img_c = img.shape

        # Calculate widht and height and paddings
        r_w = w / img_w
        r_h = h / img_h
        if r_h > r_w:
            tw = w
            th = int(r_w * img_h)
            tx1 = tx2 = 0
            ty1 = int((h - th) / 2)
            ty2 = h - th - ty1
            self.ration = r_w
        else:
            tw = int(r_h * img_w)
            th = h
            tx1 = int((w - tw) / 2)
            tx2 = w - tw - tx1
            ty1 = ty2 = 0
            self.ration = r_h

        # Resize long
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
        # pad
        padded_img_bgr = cv2.copyMakeBorder(
            img, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (114, 114, 114)
        )
        # cv2.imwrite(os.path.join(
        #     save_path, "padded_img_bgr_center.jpg"), padded_img_bgr)
        # BGR => RGB
        padded_img_rgb = cv2.cvtColor(padded_img_bgr, cv2.COLOR_BGR2RGB)
        # to tensor
        padded_img_rgb_data = padded_img_rgb.astype(np.float32)
        # Normalize to [0,1]
        padded_img_rgb_data /= 255.0
        # HWC to CHW format:
        padded_img_rgb_data = np.transpose(padded_img_rgb_data, [2, 0, 1])
        # CHW to NCHW format
        padded_img_rgb_data = np.expand_dims(padded_img_rgb_data, axis=0)
        # Convert the image to row-major order, also known as "C order":
        padded_img_rgb_data = np.ascontiguousarray(padded_img_rgb_data)
        # np.save('np_input_center', padded_img_rgb_data)
        return padded_img_rgb_data, (min(r_w, r_h), tx1, ty1)



    @staticmethod
    def _make_grid(nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype("float")

    def predict(self, data):

        z = []  # inference output


        output = self.output.asnumpy(self.output_shape) * self.output_scale

        input = {self.input_name: np.array(data, dtype=np.float32)}
        output = self.net.process(self.graph_name, input)
        output = list(output.values())[0]


        return output


    def inference(self, frame):

        if not isinstance(frame, type(None)):
            img, (ratio, tx1, ty1) = self.preprocess(frame, 3, self.input_h, self.input_w)


            dets = self.predict(img)

            output = self.postprocess(dets)
            
            result = []
            for det in output[0]:
                # if det[5] in [0, 2, 3, 5]: # filter by classes: [person, car, moterbike, bus]
                if det[5] in [0, 2]: # filter by classes: [person, car]
                    box = det[:4]
                    # scale to the origin image
                    left = int((box[0] - tx1) / ratio)
                    top = int((box[1] - ty1) / ratio)
                    right = int((box[2] - tx1) / ratio)
                    bottom = int((box[3] - ty1) / ratio)
                    result.append([[(left, top), (right, bottom)], det[4], int(det[5])])

            return result

    def postprocess(self, outs, max_wh=7680):
        bs = outs.shape[0]
        output = [np.zeros((0, 6))] * bs
        xc = outs[..., 4] > self.confThreshold
        for xi, x in enumerate(outs):
            x = x[xc[xi]]
            if not x.shape[0]:
                continue
            # Compute conf
            x[:, 5:] *= x[:, 4:5]
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])
            conf = x[:, 5:].max(1)
            j = x[:, 5:].argmax(1)
            x = np.concatenate((box, conf.reshape(-1, 1), j.reshape(-1, 1)), 1)[conf > self.confThreshold]
            c = x[:, 5:6] * max_wh  # classes
            boxes = x[:, :4] + c.reshape(-1, 1)
            scores = x[:, 4]
            i = nms_np(boxes, scores, self.nmsThreshold)
            output[xi] = x[i]

        return output