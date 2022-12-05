import os
from tqdm import tqdm
from glob import glob
import time
from argparse import ArgumentParser
import numpy as np
from tools import load_test_data, save_images 
import sophon.sail as sail 
from engine import EngineOV

def parse_args():
    # images_path 
    parser = ArgumentParser()
    parser.add_argument('--images_path', type=str, default='dataset/real', help='Directory name of test photos')
    parser.add_argument('--style_name', type=str, default='Hayao', help='what style you want to get')
    parser.add_argument("--bmodel_path", type=str, default="./dataset/animategan.pb", help="model path")
    parser.add_argument("--output_path", type=str, default="dataset/fake", help="output path")
    parser.add_argument("--tpu_id", type=int, default=0, help="tpu id")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    model = EngineOV(args.bmodel_path, args.tpu_id)
    
    for single_image in os.listdir(args.images_path):
        
        if not single_image.endswith(('.jpg', '.png', '.bmp', '.jpeg')):
            continue
        
        image_path = os.path.join(args.images_path, single_image)
        output_path = os.path.join(args.output_path, single_image)
        image, source_size = load_test_data(image_path, (512,512))
        sample_image = np.asarray(image)
        res = model({"test": sample_image})
        save_images(res, output_path,size=source_size[::-1], photo_path=None)
        
if __name__=="__main__":
    main()
