"""
This program is part of the teaching materials for teacher Hao Xiaoli's experimental class of BJTU.

Copyright © 2021 HAO xiaoli and Yang jian.
All rights reserved.
"""

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

def freeze_graph(input_checkpoint,output_graph):
    output_node_names='add_1'
    saver = tf.train.import_meta_graph(input_checkpoint+'.meta',clear_devices=True) #+'.meta'
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess,input_checkpoint)
        output_graph_def=graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(",")
        )
        with tf.gfile.GFile(output_graph,'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph. "%len(output_graph_def.node))


input_checkpoint="./Model/stock2.model-999"
out_pb_path='./LSTM_model.pb'
freeze_graph(input_checkpoint,out_pb_path)
