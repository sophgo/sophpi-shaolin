from logging import raiseExceptions
from re import I
# from tkinter.tix import Tree
import cv2
import os
import sys
from ufw.io import *
import random
import numpy as np
import argparse
import skimage.color as sc

def convertString2Bool(str):
    if str.lower() in {'true'}:
        return True
    elif str.lower() in {'false'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

def expand_path(args):
    if args.imageset_rootfolder != '':
        args.imageset_rootfolder = os.path.realpath(args.imageset_rootfolder)
    # if args.imageset_lmdbfolder != '':
    #     args.imageset_lmdbfolder = os.path.realpath(args.imageset_lmdbfolder)
    # else:
    #     args.imageset_lmdbfolder = args.imageset_rootfolder

def gen_imagelist(args):
    pic_path = args.imageset_rootfolder
    # print(pic_path)
    file_list = []
    dir_or_files = os.listdir(pic_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(pic_path,dir_file)
        ext_name = os.path.splitext(dir_file)[-1]
        if not os.path.isdir(dir_file_path) and ext_name in ['.jpg','.png','.jpeg','.bmp','.JPEG','.JPG','.BMP']:
            file_list.append(dir_file_path)
            print(dir_file_path)
    if len(file_list) == 0:
        print(pic_path+' no pictures')
        exit(1)

    if args.shuffle:
        random.seed(3)
        random.shuffle(file_list)

    return file_list


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='convert imageset')
    parse.add_argument('--imageset_rootfolder', type=str, required=True, help = 'please setting images source path')
    parse.add_argument('--shuffle', type=convertString2Bool, default=False, help = 'shuffle order of images')
    parse.add_argument('--resize_height', type=int, required=True, help = 'target height')
    parse.add_argument('--resize_width', type=int, required=True, help = 'target width')
    parse.add_argument('--scale', type=int, required=True, help = 'upsample scale > 1, best scale = 2')
    parse.add_argument('--outlmdb_rootfolder', type=str, required=True, help = 'output lmdb path')
    args = parse.parse_args()

    scale = args.scale
    expand_path(args)
    image_list = gen_imagelist(args)

    imageset_lmdbfolder_1 = args.outlmdb_rootfolder + '/data_1.lmdb' 
    imageset_lmdbfolder_2 = args.outlmdb_rootfolder + '/data_2.lmdb' 
    lmdb_1 = LMDB_Dataset(imageset_lmdbfolder_1)
    lmdb_2 = LMDB_Dataset(imageset_lmdbfolder_2)
    print("start create lmdb")
    for image_path in image_list:
        if "HR" in image_path: continue
        print(image_path)
        img = cv2.imread(image_path)
        img = cv2.resize(img, [args.resize_width, args.resize_height])
        size = img.shape
        ypbpr = sc.rgb2ypbpr(img / 255.0)
        x_scale = cv2.resize(img, [size[1] * scale, size[0] * scale], interpolation=cv2.INTER_CUBIC)
        y, pbpr = ypbpr[..., 0], sc.rgb2ypbpr(x_scale / 255.0)[..., 1:]
        y = np.expand_dims(y, -1)

        temp = pbpr.reshape(-1, 2)
        if np.max(temp[:,0]) < 1.0e-8 or np.max(temp[:,1]) < 1.0e-8: continue

        y = np.expand_dims(y, 0)
        pbpr = np.expand_dims(pbpr, 0)

        lmdb_1.put(np.ascontiguousarray(y, dtype=np.float32))
        lmdb_2.put(np.ascontiguousarray(pbpr, dtype=np.float32))

    lmdb_1.close()
    lmdb_2.close()
    print("create lmdb success")

