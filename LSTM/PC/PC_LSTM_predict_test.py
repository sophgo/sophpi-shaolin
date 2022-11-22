#coding=gbk
"""
This program is part of the teaching materials for teacher Hao Xiaoli's experimental class of BJTU.

Copyright ? 2021 HAO xiaoli and Yang jian.
All rights reserved.
"""

import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
#设置matplotlib绘图显示中文
import matplotlib as mpl
mpl.rcParams[u'font.sans-serif'] = ['SimHei']#宋体
mpl.rcParams['axes.unicode_minus'] = False #解决负号‘-‘显示为方块的问题

#设置常量
rnn_unit=10       #隐藏层神经元
input_size=7      #输入维度
output_size=1     #输出维度
step=20      #时间步

#——————————————————导入数据——————————————————————
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

#获取测试集
def get_test_data(time_step=step,test_begin=5800):
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


#——————————————————定义神经网络变量——————————————————
weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }

#——————————————————定义神经网络——————————————————
def lstm(X):     
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']  
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    #cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    lstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    cell=tf.nn.rnn_cell.MultiRNNCell([lstm for _ in range(2)])

    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32) 
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states


#————————————————预测模型————————————————————
def prediction(time_step=step):
    start_time = time.time()
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    mean,std,test_x,test_y=get_test_data(time_step)
    test_x=np.array(test_x)
    pred,_=lstm(X)     
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint("./Model/")
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)-1):
            test_x1 = test_x[step]
            test_p = np.array([test_x1], dtype=np.float32)
            prob = sess.run(pred, feed_dict={X: test_p})
            predict = prob.reshape((-1))
            test_predict.extend(predict)

        test_y=np.array(test_y)*std[7]+mean[7]
        test_predict=np.array(test_predict)*std[7]+mean[7]
        test_y=test_y[0:len(test_predict)]
        
        #评估指标：均方误差和相关系数
        mae = mean_absolute_error(test_y, test_predict)
        R = np.mean(np.multiply((test_y - np.mean(test_y)), (test_predict - np.mean(test_predict)))) / (np.std(test_y) * np.std(test_predict))
        end_time = time.time()
        timer = end_time - start_time
        print("---------------------PC--------------------------------")
        print("相关系数R: %.2f" % R)
        print('均方误差: %.2f' % mae)
        print("time consuming: %.6f sec" % timer)
        print("-------------------------------------------------------")

        #以折线图表示结果
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(list(range(len(test_y))), test_y, color='k', label='实际值')
        plt.plot(list(range(len(test_predict))), test_predict, color='r',label='预测值')
        plt.xlabel('天数', fontsize=18)
        plt.ylabel('最高价/元', fontsize=18)
        plt.legend(ncol=2, frameon=False,fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.show()

prediction()