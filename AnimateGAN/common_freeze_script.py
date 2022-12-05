import tensorflow as tf
from tensorflow.python.framework import graph_util
import argparse
import os
from net import generator # from animategan git repo 

def parse_args():
    desc = "AnimeGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--checkpoint_dir', type=str, 
    default='checkpoint/'+'generator_Hayao_weight/',
    help='Directory name to save the checkpoints')
    return parser.parse_args()

def main():
    
    arg = parse_args()
    checkpoint_dir = arg.checkpoint_dir
    print(arg.checkpoint_dir)
    
    test_real = tf.placeholder(tf.float32, [1, None, None, 3], name='test')
    with tf.variable_scope("generator", reuse=False):
        test_generated = generator.G_net(test_real).fake   
    
    generator_var = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
    saver = tf.train.Saver(generator_var)
    with tf.Session() as sess:
        print(test_generated)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            graph = tf.get_default_graph()
            input_graph_def = graph.as_graph_def()
            output_graph = os.path.join("AnimeGAN_dynamic.pb")
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=input_graph_def,
                output_node_names=["generator/G_MODEL/Tanh"])
            
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            
            print("done")
        else:
            print("error")
    return 

if __name__=="__main__":
    main()