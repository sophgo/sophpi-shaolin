import datetime
import preprocess
from matplotlib import pyplot as plt 
import cv2
import numpy as np 
import ufwio

def preprocess_handle(img):
  preprocess_list = preprocess.get_midas_preprocess()
  input_batch = preprocess.run_preprocess({"image":img},preprocess_list)
  input_batch = input_batch['image']
  return input_batch

input_path = "../dataset/dog.jpg"
img = cv2.imread(input_path)
data = preprocess_handle(img)
batch_data = np.expand_dims(data, axis=0)

lmdb = ufwio.LMDB_Dataset('../dataset/lmdb')
lmdb.put(batch_data)
lmdb.close()