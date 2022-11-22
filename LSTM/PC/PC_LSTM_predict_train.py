#coding=gbk
"""
This program is part of the teaching materials for teacher Hao Xiaoli's experimental class of BJTU.

Copyright ? 2021 HAO xiaoli and Yang jian.
All rights reserved.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 注意在import keras/tensorflow之前
import keras.backend.tensorflow_backend as KTF

import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#keras GPU动态增长
config = tf.ConfigProto()
config.gpu_options.allow_growth=True  #不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)

#设置matplotlib绘图显示中文
import matplotlib as mpl
mpl.rcParams[u'font.sans-serif'] = ['SimHei']#宋体
mpl.rcParams['axes.unicode_minus'] = False #解决负号‘-‘显示为方块的问题
from tensorflow.python.framework import graph_util

#设置常量
rnn_unit=10       #隐藏层神经元
input_size=7      #输入维度
output_size=1     #输出维度
lr=0.0006         #学习率
step=20      #时间步

#——————————————————导入数据——————————————————————
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

#获取训练集
def get_train_data(batch_size=60,time_step=step,train_begin=0,train_end=5800):
    batch_index=[]
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集 
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:7]
       y=normalized_train_data[i:i+time_step,7,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y

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


#——————————————————训练模型——————————————————
def train_lstm(batch_size=80,time_step=step,train_begin=2000,train_end=5800):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    pred,_=lstm(X)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    #module_file = tf.train.latest_checkpoint('./Model/')#初次训练需要注释掉

    train_loss=[]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) #初始化所有变量
        #saver.restore(sess, module_file)#初次训练需要注释掉
        #重复训练次数
        for i in range(1000):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print(i,loss_)
            train_loss.append(loss_)
        print("保存模型：",saver.save(sess,'./Model/stock2.model',global_step=i))

    #绘制Train loss
    plt.plot(train_loss,label='train_loss')
    plt.legend()
    plt.show()  

train_lstm()