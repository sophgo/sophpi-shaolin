import os
import warnings

import cv2
import numpy as np
# from PIL import Image

from preprocess import *
from postprocess import *
from utils import _xywh2xyxy
from utils import _box2cs



def _inference_single_pose_model(model,
                                 img_or_path,
                                 bboxes,
                                 dataset='TopDownCocoDataset',
                                 dataset_info=None,
                                 return_heatmap=False):
    """Inference human bounding boxes.

    Note:
        - num_bboxes: N
        - num_keypoints: K

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (str | np.ndarray): Image filename or loaded image.
        bboxes (list | np.ndarray): All bounding boxes (with scores),
            shaped (N, 4) or (N, 5). (left, top, width, height, [score])
            where N is number of bounding boxes.
        outputs (list[str] | tuple[str]): Names of layers whose output is
            to be returned, default: None

    Returns:
        ndarray[NxKx3]: Predicted pose x, y, score.
        heatmap[N, K, H, W]: Model output heatmap.
    """

    assert len(bboxes[0]) in [4, 5]


    batch_data = []
    for bbox in bboxes:
        center, scale = _box2cs(bbox)

        # prepare data
        data = {
            'img_or_path':
            img_or_path,
            'center':
            center,
            'scale':
            scale,
            'bbox_score':
            bbox[4] if len(bbox) == 5 else 1,
            'bbox_id':
            0,  # need to be assigned if batch_size > 1
            'dataset':
            "coco",
            'joints_3d':
            np.zeros((17,3), dtype=np.float32),
            'joints_3d_visible':
            np.zeros((17,3), dtype=np.float32),
            'rotation':
            0,
            'ann_info': {
                'image_size': np.array([192, 256]),
                'num_joints': 17,
                'flip_pairs': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
            }
        }
        # data = test_pipeline(data)
        data = preprocess(data)
        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=1)

    batch_data['img_metas'] = [
        img_metas[0] for img_metas in batch_data['img_metas'].data
    ]
    # forward the model
    [net, graph_name, input_name, output_name] = model
    batch_size = net.get_input_shape(graph_name, input_name)[0]
    true_batch = batch_data['img'].shape[0]
    batch_data['img'] = batch_data['img'].numpy()
    if true_batch > batch_size:
        output_heatmap = np.zeros((true_batch, 17, 64, 48))
        remainder = true_batch % batch_size
        quotient = true_batch // batch_size
        for i in range(quotient):
            input_data = {input_name: batch_data['img'][i*batch_size:(i+1)*batch_size]}
            output_heatmap[i*batch_size:(i+1)*batch_size] = net.process(graph_name, input_data)[output_name]
        if remainder != 0:
            input_data = np.zeros(batch_size, 3, 256, 192)
            diff = batch_size - remainder
            input_data[0:remainder] = batch_data['img'][(quotient-1)*batch_size:true_batch]
            for i in range(diff):
                input_data[remainder+i] = batch_data['img'][0]
            input_data = { input_name: input_data}
            output_heatmap[(quotient-1)*batch_size:true_batch] = net.process(graph_name, input_data)[output_name][0:true_batch]
    elif true_batch < batch_size:
        input_data = np.zeros(batch_size, 3, 256, 192)
        diff = batch_size - true_batch
        input_data[0:true_batch] = batch_data['img']
        for i in range(diff):
            input_data[true_batch+i] = batch_data['img'][0]
        input_data = { input_name: input_data}
        output_heatmap = net.process(graph_name, input_data)[output_name][0:true_batch]
    else:
        input_data = {input_name: batch_data['img']}
        output_heatmap = net.process(graph_name, input_data)[output_name]
    _, _, img_height, img_width = batch_data['img'].shape
    result = decode(batch_data['img_metas'], output_heatmap, img_size=[img_height, img_width])
    return result['preds'], None


def inference_top_down_pose_model(model,
                                  img_or_path,
                                  person_results=None,
                                  bbox_thr=None,
                                  format='xywh',
                                  return_heatmap=False,
                                  outputs=None):
    """Inference a single image with a list of person bounding boxes.

    Note:
        - num_people: P
        - num_keypoints: K
        - bbox height: H
        - bbox width: W

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (str| np.ndarray): Image filename or loaded image.
        person_results (list(dict), optional): a list of detected persons that
            contains ``bbox`` and/or ``track_id``:

            - ``bbox`` (4, ) or (5, ): The person bounding box, which contains
                4 box coordinates (and score).
            - ``track_id`` (int): The unique id for each human instance. If
                not provided, a dummy person result with a bbox covering
                the entire image will be used. Default: None.
        bbox_thr (float | None): Threshold for bounding boxes. Only bboxes
            with higher scores will be fed into the pose detector.
            If bbox_thr is None, all boxes will be used.
        format (str): bbox format ('xyxy' | 'xywh'). Default: 'xywh'.

            - `xyxy` means (left, top, right, bottom),
            - `xywh` means (left, top, width, height).
        return_heatmap (bool) : Flag to return heatmap, default: False
        outputs (list(str) | tuple(str)) : Names of layers whose outputs
            need to be returned. Default: None.

    Returns:
        tuple:
        - pose_results (list[dict]): The bbox & pose info. \
            Each item in the list is a dictionary, \
            containing the bbox: (left, top, right, bottom, [score]) \
            and the pose (ndarray[Kx3]): x, y, score.
        - returned_outputs (list[dict[np.ndarray[N, K, H, W] | \
            Tensor[N, K, H, W]]]): \
            Output feature maps from layers specified in `outputs`. \
            Includes 'heatmap' if `return_heatmap` is True.
    """

    # only two kinds of bbox format is supported.
    assert format in ['xyxy', 'xywh']

    pose_results = []
    returned_outputs = []

    if person_results is None:
        # create dummy person results
        if isinstance(img_or_path, str):
            width, height = Image.open(img_or_path).size
        else:
            height, width = img_or_path.shape[:2]
        person_results = [{'bbox': np.array([0, 0, width, height])}]

    if len(person_results) == 0:
        return pose_results, returned_outputs

    # Change for-loop preprocess each bbox to preprocess all bboxes at once.
    bboxes = np.array([box['bbox'] for box in person_results])

    # Select bboxes by score threshold
    if bbox_thr is not None:
        assert bboxes.shape[1] == 5
        valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
        bboxes = bboxes[valid_idx]
        person_results = [person_results[i] for i in valid_idx]

    if format == 'xyxy':
        bboxes_xyxy = bboxes
        bboxes_xywh = _xyxy2xywh(bboxes)
    else:
        # format is already 'xywh'
        bboxes_xywh = bboxes
        bboxes_xyxy = _xywh2xyxy(bboxes)

    # if bbox_thr remove all bounding box
    if len(bboxes_xywh) == 0:
        return [], []

    with OutputHook(model, outputs=outputs, as_tensor=False) as h:
        # poses is results['pred'] # N x 17x 3
        poses, heatmap = _inference_single_pose_model(
            model,
            img_or_path,
            bboxes_xywh,
            return_heatmap=return_heatmap)

        if return_heatmap:
            h.layer_outputs['heatmap'] = heatmap

        returned_outputs.append(h.layer_outputs)

    assert len(poses) == len(person_results), print(
        len(poses), len(person_results), len(bboxes_xyxy))
    for pose, person_result, bbox_xyxy in zip(poses, person_results,
                                              bboxes_xyxy):
        pose_result = person_result.copy()
        pose_result['keypoints'] = pose
        pose_result['bbox'] = bbox_xyxy
        pose_results.append(pose_result)

    return pose_results, returned_outputs


