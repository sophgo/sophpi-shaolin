'''
This program is part of the teaching materials for teacher Hao Xiaoli's experimental class of BJTU.

Copyright © 2021 HAO xiaoli and CHANG tianxing.
All rights reserved.
'''
import cv2
import time
import argparse
import numpy as np
from PIL import Image
import sophon.sail as sail


def preprocess(img):
    image = Image.fromarray(img)
    # 请基于 train.py 的前处理填充前训练代码
    resized_img = np.array(image.resize((224, 224), resample=2))
    out = np.array(resized_img / 255., dtype=np.float32)
    return out.transpose((2, 0, 1))


def postprocess(output):
    cat_prob = output[:, 0]
    dog_prob = output[:, 1]
    if (cat_prob > dog_prob):
        return 'cat'
    else:
        return 'dog'


def infer(bmodel_path, img_path, tpu_id=0):
    start_time = time.time()
    img = cv2.imread(img_path)
    img_array = preprocess(img)

    # 请补充基于 sail 的推理代码，推理结果为字典，字典的key值分别是输出节点名
    # 推理结果保存于 prediction 变量，输出节点名列表 保存于output_names
    net = sail.Engine(bmodel_path, tpu_id, sail.IOMode.SYSIO)
    graph_name = net.get_graph_names()[0]  # get net name
    input_names = net.get_input_names(graph_name)  # [input_names], 输入节点名列表
    output_names = net.get_output_names(graph_name)  # [output_names],输出节点名列表
    input_data = {input_names[0]: np.expand_dims(img_array, axis=0)}

    # 模型推理，推理结果为字典，字典的key值分别是输出节点名
    prediction = net.process(graph_name, input_data)

    end_time = time.time()
    timer = end_time - start_time
    print("-" * 66)
    print('The output value of CATS: %.5f' % prediction[output_names[0]][:, 0])
    print('The output value of DOGS: %.5f' % prediction[output_names[0]][:, 1])
    print("Time consuming: %.5f sec" % timer)
    print("-" * 66)
    res = postprocess(prediction[output_names[0]])
    print("The result is {}".format(res))
    return res


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='for sail py test')
    PARSER.add_argument('--bmodel', default='compilation.bmodel')
    PARSER.add_argument('--img_path', default='data/test.0.jpg')
    PARSER.add_argument('--tpu_id', default=0, type=int, required=False)
    ARGS = PARSER.parse_args()
    infer(ARGS.bmodel, ARGS.img_path, ARGS.tpu_id)
