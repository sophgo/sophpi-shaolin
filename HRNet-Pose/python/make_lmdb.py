import os
import warnings
from argparse import ArgumentParser
from utils import _box2cs
from PIL import Image
from utils import COCO, _xywh2xyxy, _xyxy2xywh
from preprocess import *
import ufwio 

def handle_each_image(img_or_path,  
                     dataset='TopDownCocoDataset', 
                     dataset_info=None,
                     person_results=None):
    
    if person_results is None:
        # create dummy person results
        if isinstance(img_or_path, str):
            width, height = Image.open(img_or_path).size
        else:
            height, width = img_or_path.shape[:2]
        person_results = [{'bbox': np.array([0, 0, width, height])}]    
    # Change for-loop preprocess each bbox to preprocess all bboxes at once.
    bboxes = np.array([box['bbox'] for box in person_results])
    
    bbox_thr=None
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
    
    bboxes = bboxes_xywh
    
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
    true_batch = batch_data['img'].shape[0]
    batch_data['img'] = batch_data['img'].numpy()
    return batch_data['img']

def main():
    parser = ArgumentParser()
    parser.add_argument('--img-root', type=str, default='../dataset/', help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default='../dataset/test_coco.json',
        help='Json file containing image info.')

    args = parser.parse_args()
    coco = COCO(args.json_file)
    img_keys = list(coco.imgs.keys())

    lmdb = ufwio.LMDB_Dataset("../dataset/lmdb")

    for i in range(len(img_keys)):
        # get bounding box annotations
        image_id = img_keys[i]
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(args.img_root, image['file_name'])
        ann_ids = coco.getAnnIds(image_id)

        # make person bounding boxes
        person_results = []
        for ann_id in ann_ids:
            person = {}
            ann = coco.anns[ann_id]
            # bbox format is 'xywh'
            person['bbox'] = ann['bbox']
            person_results.append(person)

        data = handle_each_image(image_name, person_results=person_results)
        lmdb.put(data)
    lmdb.close()
    
if __name__ == '__main__':
    main()
