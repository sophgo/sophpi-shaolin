'''
This program is part of the teaching materials for teacher Hao Xiaoli's experimental class of BJTU.

Copyright © 2021 HAO xiaoli and CHANG tianxing.
All rights reserved.
'''
# tensorflow1.15
# !--*-- coding: utf-8 --*--
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import colorsys
import time
import os
import cv2 as cv
import random

classes_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']

model_path = "./"
frozen_pb_file = os.path.join(model_path,
                              './mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb')

# 加载 Graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(frozen_pb_file, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# sess 配置
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def preprocess(img_file):
    img = cv.imread(img_file)
    if img.shape[0] < img.shape[1]:
        H = int(img.shape[0] * 1920 / img.shape[1])
        img_array = cv.resize(img, (1920, H), cv.INTER_NEAREST)
        img_padding = np.pad(img_array, ((0, 1920 - H), (0, 0), (0, 0)), 'mean')
    else:
        W = int(img.shape[1] * 1920 / img.shape[0])
        img_array = cv.resize(img, (W, 1920), cv.INTER_NEAREST)
        img_padding = np.pad(img_array, ((0, 0), (0, 1920 - W), (0, 0)))
    return img_padding, img_array.shape[0], img_array.shape[1]


def noresize(img_file):
    img = cv.imread(img_file)
    H, W = img.shape[0], img.shape[1]
    return img, H, W


def detect(img_array):
    img_array_expanded = np.expand_dims(img_array, axis=0)

    results = {}
    with detection_graph.as_default():
        with tf.Session(config=config) as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = \
                {output.name for op in ops for output in op.outputs}

            tensor_dict = {}
            for key in ['num_detections',
                        'detection_boxes',
                        'detection_scores',
                        'detection_classes',
                        'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph(). \
                        get_tensor_by_name(tensor_name)
            image_tensor = tf.get_default_graph(). \
                get_tensor_by_name('image_tensor:0')

            start = time.time()
            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: img_array_expanded})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            results['num_detections'] = int(output_dict['num_detections'][0])
            results['classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            results['boxes'] = output_dict['detection_boxes'][0]
            results['scores'] = output_dict['detection_scores'][0]
            results['masks'] = output_dict['detection_masks'][0]
    return results, start


def vis_res(img_padding, result, start, score_threshold=0.66, alpha=0.3):
    # Visualize detected bounding boxes and masks.
    num_detections = int(result['num_detections'])
    classes = result['classes']
    boxes = result['boxes']
    scores = result['scores']
    masks = result['masks']

    img_height, img_width = img_padding.shape[:2]
    colors = random_colors(num_detections)
    for idx in range(num_detections):
        color = np.asanyarray(colors[idx]) * 255
        color_tuple = (int(color[0]), int(color[1]), int(color[2]))
        mask = masks[idx]
        score = float(scores[idx])
        if score > score_threshold:
            class_id = int(classes[idx])

            # box
            box = boxes[idx] * np.array(
                [img_height, img_width, img_height, img_width])
            top, left, bottom, right = box.astype("int")
            left = max(0, min(left, img_width - 1))
            top = max(0, min(top, img_height - 1))
            right = max(0, min(right, img_width - 1))
            bottom = max(0, min(bottom, img_height - 1))

            # Draw a bounding box.
            cv.rectangle(img_padding, (left, top), (right, bottom), color_tuple, 2)

            label = "{}: {:.2f}".format(classes_names[class_id], score)

            # Display the label at the top of the bounding box
            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            # Draw top label box
            cv.rectangle(img_padding, (left, top - round(1.5 * labelSize[1])),
                         (left + round(1.5 * labelSize[0]), top + baseLine),
                         (255, 255, 255), cv.FILLED)
            cv.putText(img_padding, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

            # resize mask
            mask_resize = cv.resize(mask, (right - left + 1, bottom - top + 1))
            roi_mask = (mask_resize > 0.3)
            mask_region = img_padding[top:bottom + 1, left:right + 1][roi_mask]

            # Draw mask
            img_padding[top:bottom + 1, left:right + 1][roi_mask] = \
                ([alpha * color[0], alpha * color[1], alpha * color[2]]
                 + (1 - alpha) * mask_region).astype(np.uint8)

            # Draw the contours on the image
            contours, hierarchy= cv.findContours(roi_mask.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(img_padding[top:bottom + 1, left:right + 1], contours,
                            -1, color_tuple, 1, cv.LINE_8, hierarchy, 100)
    end = time.time()
    print("cost time: {:.4f} s".format(end - start))
    cv.imwrite("maskrcnn_out_{}".format(img_file[16:]), img_padding[:H, :W, :])
    # [:,:,::-1] bgr->rgb
    plt.imshow(img_padding[:, :, ::-1])
    plt.title("TensorFlow Mask RCNN-Inception_v2_coco")
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    img_file = "./Mask_RCNN_SE5/6.png"
    img_padding, H, W = preprocess(img_file)
    # img_padding, H, W = noresize(img_file)

    results, start_time = detect(img_padding)
    vis_res(img_padding, results, start_time)
    print("Done")
