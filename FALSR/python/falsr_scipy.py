import argparse
from email.mime import image
import cv2
import numpy as np
import sophon.sail as sail
import time
import skimage.color as sc
import os

class Falsr(object):
    def __init__(self, bmodel_path, tpu_id, sr_scale):
        #load bmodel and get model information
        self.net = sail.Engine(bmodel_path, tpu_id, sail.IOMode.SYSIO)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name_y = self.net.get_input_names(self.graph_name)[0]
        self.input_name_pbpr = self.net.get_input_names(self.graph_name)[1]
        self.output_name = self.net.get_output_names(self.graph_name)[0]
        self.input_shape_y = self.net.get_input_shape(
            self.graph_name, self.input_name_y)
        self.input_shape_pbpr = self.net.get_input_shape(
            self.graph_name, self.input_name_pbpr)

        #get input and output shape
        self.input_y_w = int(self.input_shape_y[-2])
        self.input_y_h = int(self.input_shape_y[-3])
        self.input_pbpr_w = int(self.input_shape_pbpr[-2])
        self.input_pbpr_h = int(self.input_shape_pbpr[-3])
        self.output_w = self.input_pbpr_w
        self.output_h = self.input_pbpr_h
        self.img_list = []
        self.sr_scale = sr_scale
        assert (self.input_pbpr_w == self.input_y_w * self.sr_scale and self.input_pbpr_h == self.input_y_h * self.sr_scale)

    #preprocess: 1.convert rgb to ypbpr   2.split ypbpr channels to y channel and pbpr channel
    def preprocess(self, img):
        if img.shape[0] != self.input_y_h or img.shape[1] != self.input_y_w:
            cv2.resize(img, (self.input_y_w, self.input_y_h))
        ypbpr = sc.rgb2ypbpr(img / 255.0)
        x_scale = cv2.resize(img, (self.input_pbpr_w, self.input_pbpr_h), interpolation=cv2.INTER_CUBIC)
        y, pbpr = ypbpr[..., 0], sc.rgb2ypbpr(x_scale / 255)[..., 1:]
        y = np.expand_dims(y, -1)
        y = np.expand_dims(y, 0)
        pbpr = np.expand_dims(pbpr, 0)
        return y, pbpr

    #bmodel forward
    def pridict(self, input_y, input_pbpr):
        inputs = {self.input_name_y: np.array(input_y, dtype=np.float32), self.input_name_pbpr: np.array(input_pbpr, dtype=np.float32)}
        output = self.net.process(self.graph_name, inputs)
        return list(output.values())[0]
    
    #postprocess model output
    def postprocess(self, model_out):
        model_out = model_out[0]
        model_out *= 255
        model_out = model_out.astype(np.uint8)
        return model_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Demo of FALSR use python")

    parser.add_argument('--bmodel',
                        type=str,
                        default="../data/models/falsr_fp32.bmodel",
                        required=False,
                        help='bmodel file path.')
    
    parser.add_argument('--video_path',
                        type=str,
                        default="../data/videos/test_270x480.mp4",
                        required=False,
                        help='input video file path.')

    parser.add_argument('--iamges_path',
                        default="../data/images",
                        type=str,
                        required=False,
                        help='output video path')
    
    parser.add_argument('--tpu_id',
                        default=0,
                        type=int,
                        required=False,
                        help='tpu dev id(0,1,2,...).')

    parser.add_argument('--sr_scale',
                        default=2,
                        type=int,
                        required=False,
                        help='super resolution upsample scale')
    
    parser.add_argument('--out_path',
                        default="./out.avi",
                        type=str,
                        required=False,
                        help='output video path')
    
    parser.add_argument('--input_type',
                        default='video',
                        type=str,
                        required=False,
                        help='video or images')

    opt = parser.parse_args()

    t_start = time.time()
    falsr = Falsr(bmodel_path = opt.bmodel,
                  tpu_id = opt.tpu_id,
                  sr_scale = opt.sr_scale)

    #load images
    if opt.input_type == 'images':
        images = os.listdir(opt.iamges_path)
        for i in range(len(images)):
            img = cv2.imread(opt.iamges_path + '/' + images[i])
            falsr.img_list.append(img)
        
        for j in range(len(falsr.img_list)):
            img = falsr.img_list[j]
            y, pbpr = falsr.preprocess(img)
            model_out = falsr.pridict(y, pbpr)
            out_img = falsr.postprocess(model_out)
            cv2.imwrite('./out_images' + str(j) + '.png', out_img)
         

    #load video
    elif opt.input_type == 'video':
        capture = cv2.VideoCapture(opt.video_path)
        if capture.isOpened():
            while True:
                ret, img=capture.read()          
                if not ret:break
                falsr.img_list.append(img)    
        else:
            print('open video failed !!!')

        #init out video writer
        video_fps = 25
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        videoWriter = cv2.VideoWriter(opt.out_path, fourcc, video_fps, (falsr.output_w, falsr.output_h))
        t_init = time.time()
        print("init bmodel and video writer time: ", t_init-t_start)
    
        #for each frame do super resoluton and write into video
        total_pre_time, total_pridict_time,  total_post_time = 0, 0, 0
        for i in range(len(falsr.img_list)):
            img = falsr.img_list[i]
            t_sr_start = time.time()
            y, pbpr = falsr.preprocess(img)
            t_pre = time.time()
            total_pre_time += t_pre - t_sr_start

            model_out = falsr.pridict(y, pbpr)
            t_pridict = time.time()
            total_pridict_time += t_pridict - t_pre

            out_img = falsr.postprocess(model_out)
            t_post = time.time()
            total_post_time += t_post - t_pridict
            videoWriter.write(out_img)
            print("frame {} cost time: ".format(i), t_post-t_sr_start)
        videoWriter.release()

    else:
        print("unknown input type")
    print("\naverage preprocess time: ", total_pre_time/len(falsr.img_list))
    print("average model forward time: ", total_pridict_time/len(falsr.img_list))
    print("average postprocess time: ", total_post_time/len(falsr.img_list))
    print("!!! video resoluton done !!!  out video saved to: ", opt.out_path)
 
