# coding:utf-8
"""
This program is part of the teaching materials for teacher Hao Xiaoli's experimental class of BJTU.

Copyright © 2021 HAO xiaoli and Yang jian.
All rights reserved.
"""

from __future__ import division
import argparse
import cv2
import time
import sophon.sail as sail
import csv
import numpy as np

#设置常量
input_size=7      #输入维度
output_size=1     #输出维度
step=20           #时间步

#导入数据
def import_data():
    with open('./dataset/dataset_2.csv', 'r') as csvfile:
         reader = csv.reader(csvfile)
         data = []
         i = 0
         for x in reader:
             if i != 0:
                x = x[2:10]  # 取第3-9列
                x = [float(k) for k in x]
                data.append(x)
             i = i + 1
         data = np.array(data)  
    return data

#预处理：数据标准化、提取并整理测试集
def preprocess_test_data(time_step=step,test_begin=5800):
    data=import_data()
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample 
    test_x,test_y=[],[]  
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:7]
       y=normalized_test_data[i*time_step:(i+1)*time_step,7]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:7]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,7]).tolist())
    return mean,std,test_x,test_y

#后处理：反标准化、计算模型评估指标  
def postpress_prediction(std,mean,test_y,test_predict):
    test_y = np.array(test_y) * std[7] + mean[7]
    test_predict = np.array(test_predict) * std[7] + mean[7]
    test_y=test_y[0:len(test_predict)]

    # 评估指标：均方误差和相关系数
    mae = np.mean(abs(test_y-test_predict))
    R = np.mean(np.multiply((test_y - np.mean(test_y)), (test_predict - np.mean(test_predict)))) / (
                 np.std(test_y) * np.std(test_predict))
    return mae,R   

#预测模型推理主函数
def main(time_step=step):
    start_time = time.time()  #开始时间
    #预处理：调用预处理函数
    mean,std,test_x,test_y=preprocess_test_data(time_step)  
    test_x=np.array(test_x)
   
    # sail core (inference)
    net = sail.Engine(ARGS.ir_path, ARGS.tpu_id, sail.IOMode.SYSIO)  #加载bmodel
    graph_name = net.get_graph_names()[0]  #获取网络名字
    input_names = net.get_input_names(graph_name)  #获取网络输入名字

    test_predict = []
    for step in range(len(test_x) - 1):
        test_x1 = test_x[step]
        data = np.array([test_x1], dtype=np.float32)
        data=np.ascontiguousarray(data)
        input_data = {input_names[0]: data}
        prob = net.process(graph_name, input_data)  #运行推理网络
        for key in prob:
            prob=prob[key]
        predict = prob.reshape((-1))
        test_predict.extend(predict)                

    #后处理：调用后处理函数
    mae,R=postpress_prediction(std,mean,test_y,test_predict)
    
    end_time = time.time()  #结束时间
    timer = end_time - start_time

    print("---------------------SE5端推理结果----------------------")
    print("相关系数R: %.2f" % R)
    print('均方误差mae: %.2f' % mae)
    print("time consuming: %.6f sec" %timer) 
    print("-------------------------------------------------------")

if __name__ == '__main__':
  PARSER = argparse.ArgumentParser(description='for sail py test')
  PARSER.add_argument('--ir_path', default='./lstm_out/compilation.bmodel')
  PARSER.add_argument('--tpu_id', default=0, type=int, required=False)
  ARGS = PARSER.parse_args()
  main()
