import base64
import cv2
import numpy as np
import threading
import os
import torch

import sophon.sail as sail 

from utils.timer import Timer
from utils.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.align import warp_and_crop_face

import iou_detect_test

_t = {'forward_pass': Timer(), 'misc': Timer()}

cfg = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

def preprocess(img,input_size):
    height = img.shape[0]
    width = img.shape[1]
    size = max(width, height)
    img_padding = cv2.copyMakeBorder(img, 0, size - height, 0, size - width, cv2.BORDER_CONSTANT, (0, 0, 0))
    image_array = cv2.resize(img_padding,(input_size,input_size))
    image_array = image_array.transpose(2, 0, 1)
    tensor = np.expand_dims(image_array, axis=0)
    tensor = np.ascontiguousarray(tensor)
    return tensor,height,width

def facepreprocess(img):

    image_array = cv2.resize(img,(112,112))
    image_array = image_array.transpose(2, 0, 1)
    tensor = np.expand_dims(image_array, axis=0)
    tensor = np.ascontiguousarray(tensor)
    return tensor

class face():
    def __init__(self,faceDP,faceRP):
        self.face_net = sail.Engine(faceRP,  1 , sail.IOMode.SYSIO)
        self.face_graph_name = self.face_net.get_graph_names()[0] 
        self.face_input_names = self.face_net.get_input_names(self.face_graph_name)[0] 

        self.net = sail.Engine(faceDP,  1 , sail.IOMode.SYSIO)
        self.graph_name = self.net.get_graph_names()[0] 
        self.input_names = self.net.get_input_names(self.graph_name)[0]  
        print('bmodel init sucess!')
        self.resize = 640
        self.img_input = 640
        self.confidence_threshold = 0.02
        self.nms_threshold = 0.4
        self.vis_thres = 0.5

    def getFaceFeature(self,img,pts):
        faceimg = warp_and_crop_face(img,pts)
        output = self.face_net.process(self.face_graph_name, {self.face_input_names: faceimg})
        return output

    # @classmethod 
    def getFeature(self,img,iou_test):
        dets,ldms,imgdraw,textPos = self.detect(img,iou_test)
        output=[]
        _t['forward_pass'].tic()
        for det,ldm in zip(dets,ldms):
            if det[4] < self.vis_thres:
                continue
            ldm.resize([5,2])
            faceimg = warp_and_crop_face(img,ldm)
            faceinput = facepreprocess(faceimg)
            feature_nmodel = self.face_net.process(self.face_graph_name, {self.face_input_names: faceinput})
            # print('-*100')
            output.append(torch.tensor(feature_nmodel['24']))
        _t['forward_pass'].toc()
        print('face_recog: {} faces, forward_pass_time: {:.4f}ms '.format(len(dets),1000 * _t['forward_pass'].average_time))
        return output,imgdraw,textPos 


    def detect(self,img_src,iou_test):
        img,im_height,im_width = preprocess(img_src,self.img_input)
        scale = max(im_height,im_width)/self.img_input
        # print(input_array.shape)
        # while(1):
        _t['forward_pass'].tic()
        output = self.net.process(self.graph_name, {self.input_names: img})
        landms = torch.Tensor(output['111'])
        loc = torch.Tensor(output['87'])
        conf = torch.Tensor(output['116'])
        _t['forward_pass'].toc()
        priorbox = PriorBox(cfg, image_size=(self.img_input, self.img_input))
        priors = priorbox.forward()
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        scale_box = torch.Tensor([img_src.shape[1], img_src.shape[0], img_src.shape[1], img_src.shape[0]])
        boxes = boxes * scale *self.img_input
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        dotldms =  torch.Tensor([img_src.shape[1],img_src.shape[0],img_src.shape[1],img_src.shape[0],img_src.shape[1],img_src.shape[0],img_src.shape[1],img_src.shape[0],img_src.shape[1],img_src.shape[0]])

        landms = landms * scale * self.img_input
        landms = landms.cpu().numpy()
        print('face_detect forward_pass_time: {:.4f}ms '.format(1000 * _t['forward_pass'].average_time))
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        # order = scores.argsort()[::-1][:args.top_k]
        order = scores.argsort()[::-1]
        order=np.ascontiguousarray(order, dtype=np.int)
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)

        dets = dets[keep, :]
        landms = landms[keep]
        dets_ = np.concatenate((dets, landms), axis=1)
        textPos=[]
        # print(dets_)
        for b in dets_:

            if b[4] < self.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_src, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            if iou_test == 1:
                with open(detect_result_filename,'a') as f:
                    f.write('%d,%d,%d,%d\n' % (b[0], b[1], b[2], b[3]))
                    #f.write('%d' % )
            cx = b[0]
            cy = b[1] + 12
            textPos.append([cx,cy])
            # cv2.putText(img_src, text, (cx, cy),
            #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_src, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_src, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_src, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_src, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_src, (b[13], b[14]), 1, (255, 0, 0), 4)
        detsR = []
        landmsR =[]
        for det,ldm in zip(dets,landms):
            if det[4] < self.vis_thres:
                continue
            detsR.append(det)
            landmsR.append(ldm)
        print('detect,textpos',textPos)
        return detsR,landmsR,img_src,textPos


    def getScore(self,a,b):
        diff = a - b
        dist = torch.sum(torch.pow(diff, 2), dim=0)
        return dist

    def getScoreCos(self,a,b):
        v1 = a.numpy()
        v2 = b.numpy()
        angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
        return angle


def getFrame(image_data,iou_test=0):
    #解析图片数据
    # image_data = np.fromstring(img, np.uint8)
    # image_data = cv2.imdecode(img, cv2.IMREAD_COLOR)
    realFeatures, imgdraw, textPoses = face.getFeature(image_data,iou_test)
    for realFeature,textPos in zip(realFeatures,textPoses):
        print(len(featurePool))
        for idFeature in featurePool:
            score = face.getScore(realFeature.squeeze(0),featurePool[idFeature])
            print(score)
            if score <0.5:          
                print('textPos',textPos)
                cv2.putText(imgdraw, 'score:{:.1f}'.format(score), (textPos[0],textPos[1]),cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                cv2.putText(imgdraw, 'name:' + idFeature, (textPos[0],textPos[1]+10),cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0))
                break
    
    return image_data


if __name__ == "__main__":
    detect_model = '../dataset/bmodel/detect/compilation.bmodel'
    feature_model = '../dataset/bmodel/feature/compilation.bmodel'
    image_path = '../dataset/data/test/test.jpg'
    ID_path = '../dataset/data/ID'
    detect_result_filename = 'detect_result.txt'
    print('if image exists:',os.path.exists(image_path))
    frame = cv2.imread(image_path)
    frame = cv2.resize(frame, (480, 270))

    imglist = os.listdir(ID_path)
    print(imglist)
    featurePool=dict()

    # init 
    face = face(detect_model,feature_model)

    # detect id images
    for img in imglist:
        imgpath=os.path.join(ID_path,img)
        print('ID image exists:',os.path.isfile(imgpath))
        print(imgpath)
        print('*'*66)
        id_img = cv2.imread(imgpath)
        if id_img is None:
            print('read ID picture failed')
        else:
            feature,_,_ = face.getFeature(id_img,0)
            if len(feature)==1:
                idF = {img[:-4]:feature[0].squeeze(0)}
                featurePool.update(idF)


    # process 
    output = getFrame(frame,1)

    # output 
    cv2.imwrite('output.jpg',output)

    # iou test
    iou_detect_test.iou_test()