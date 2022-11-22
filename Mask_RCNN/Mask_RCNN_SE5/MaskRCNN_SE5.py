'''
This program is part of the teaching materials for teacher Hao Xiaoli's experimental class of BJTU.

Copyright © 2021 HAO xiaoli and CHANG tianxing.
All rights reserved.
'''
import numpy as np
import colorsys
import os
import cv2 as cv
import random
import argparse
import time
import sophon.sail as sail

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


def vis_res(img_array, result, H, W, score_threshold=0.6, alpha=0.3):
    # Visualize detected bounding boxes and masks.
    num_detections = int(result['num_detections'][0])
    classes = result['detection_classes'][0]
    boxes = result['detection_boxes'][0]
    scores = result['detection_scores'][0]
    masks = result['detection_masks'][0]

    img_height, img_width = img_array.shape[:2]
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
            cv.rectangle(img_array, (left, top), (right, bottom), color_tuple, 2)

            label = "{}: {:.2f}".format(classes_names[class_id], score)

            # Display the label at the top of the bounding box
            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            # Draw top label box
            cv.rectangle(img_array, (left, top - round(1.5 * labelSize[1])),
                         (left + round(1.5 * labelSize[0]), top + baseLine),
                         (255, 255, 255), cv.FILLED)
            cv.putText(img_array, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

            # resize mask
            mask_resize = cv.resize(mask, (right - left + 1, bottom - top + 1))
            roi_mask = (mask_resize > 0.3)
            mask_region = img_array[top:bottom + 1, left:right + 1][roi_mask]

            # Draw mask
            img_array[top:bottom + 1, left:right + 1][roi_mask] = \
                ([alpha * color[0], alpha * color[1], alpha * color[2]]
                 + (1 - alpha) * mask_region).astype(np.uint8)

            # Draw the contours on the image
            contours, hierarchy = cv.findContours(roi_mask.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(img_array[top:bottom + 1, left:right + 1], contours,
                            -1, color_tuple, 1, cv.LINE_8, hierarchy, 100)
    cv.imwrite("./maskrcnn_out_{}".format(ARGS.input[2:]), img_array[:H, :W, :])


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
        img_array = cv.resize(img, (1920, H))
        img_padding = np.pad(img_array, ((0, 1920 - H), (0, 0), (0, 0)), 'mean')
    else:
        W = int(img.shape[1] * 1920 / img.shape[0])
        img_array = cv.resize(img, (W, 1920))
        img_padding = np.pad(img_array, ((0, 0), (0, 1920 - W), (0, 0)), 'mean')
    return img_padding, img_array.shape[0], img_array.shape[1]


def main():
    """Program entry point.
    """
    img_padding, H, W = preprocess(ARGS.input)
    input_array = np.expand_dims(img_padding, axis=0)

    # sail core (inference)
    net = sail.Engine(ARGS.bmodel, ARGS.tpu_id, sail.IOMode.SYSIO)
    graph_name = net.get_graph_names()[0]  # get net name
    input_names = net.get_input_names(graph_name)[0]  # get input names
    output_names = net.get_output_names(graph_name)
    print('output_names=', output_names)

    input_data = {input_names: np.array(input_array, dtype=np.uint8)}
    start = time.time()
    output = net.process(graph_name, input_data)

    # postprocess maskrcnn
    vis_res(img_padding, output, H, W)
    end = time.time()
    print("-" * 66)
    print('saved as maskrcnn_out_{}'.format(ARGS.input[2:]))
    print("cost time: {:.5f} s".format(end - start))
    print("-" * 66)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='for sail py test')
    PARSER.add_argument('--bmodel', default='./Mask_RCNN_bmodel/compilation.bmodel')
    PARSER.add_argument('--input', default='./street.jpg')
    PARSER.add_argument('--tpu_id', default=0, type=int, required=False)
    ARGS = PARSER.parse_args()
    main()
