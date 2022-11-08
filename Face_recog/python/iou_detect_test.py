from ast import Gt
import cv2
import numpy as np
 
def CountIOU(RecA, RecB):
    xA = max(RecA[0], RecB[0])
    yA = max(RecA[1], RecB[1])
    xB = min(RecA[2], RecB[2])
    yB = min(RecA[3], RecB[3])
    
    # 计算交集部分面积
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # 计算预测值和真实值的面积
    RecA_Area = (RecA[2] - RecA[0] + 1) * (RecA[3] - RecA[1] + 1)
    RecB_Area = (RecB[2] - RecB[0] + 1) * (RecB[3] - RecB[1] + 1)
    
    # 计算IOU
    iou = interArea / float(RecA_Area + RecB_Area - interArea)
    return iou

def read_txt(txtfile):
    return np.genfromtxt(txtfile,dtype=[int,int,int,int],delimiter=',')

def iou_test():
    gt = read_txt("detect_groundtruth.txt")   # read groundtruth to numpy
    result = read_txt("detect_result.txt")  #
    ans = 0

    for i in range(gt.shape[0]):
        RecA = gt[i].tolist()
        RecB = result[i].tolist()
        IOU = CountIOU(RecA, RecB)
        font = cv2.FONT_HERSHEY_SIMPLEX
        ans += IOU
    ans /= gt.shape[0]
    with open('./accuracy_result.txt','w') as f:
        f.write('%f' % ans)
    