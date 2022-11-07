from torchvision.transforms import functional 

from utils import *
from utils import _to_tensor
from utils import DataContainer as DC
import cv2


def load_image(results):
    results['image_file'] = results['img_or_path']
    img = cv2.imread(results['img_or_path'])
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    results['img'] = img
    return results

def top_down_affine(results):
    image_size = results['ann_info']['image_size']

    img = results['img']
    joints_3d = results['joints_3d']
    joints_3d_visible = results['joints_3d_visible']
    c = results['center']
    s = results['scale']
    r = results['rotation']
    trans = get_affine_transform(c, s, r, image_size)
    if not isinstance(img, list):
        img = cv2.warpAffine(
            img,
            trans, (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)
    else:
        img = [
            cv2.warpAffine(
                i,
                trans, (int(image_size[0]), int(image_size[1])),
                flags=cv2.INTER_LINEAR) for i in img
        ]
    for i in range(results['ann_info']['num_joints']):
        if joints_3d_visible[i, 0] > 0.0:
            joints_3d[i,
                        0:2] = affine_transform(joints_3d[i, 0:2], trans)
    results['img'] = img
    results['joints_3d'] = joints_3d
    results['joints_3d_visible'] = joints_3d_visible

    return results


def to_tensor(results):
    if isinstance(results['img'], (list, tuple)):
        results['img'] = [_to_tensor(img) for img in results['img']]
    else:
        results['img'] = _to_tensor(results['img'])

    return results

def normalize_tensor(results, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if isinstance(results['img'], (list, tuple)):
        results['img'] = [
            functional.normalize(img, mean=mean, std=std, inplace=True)
            for img in results['img']
        ]
    else:
        results['img'] = functional.normalize(
            results['img'], mean=mean, std=std, inplace=True)

    return results

def collect(results, keys=['img'], meta_name='img_metas', meta_keys=['image_file', 'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs']):
    if 'ann_info' in results:
        results.update(results['ann_info'])

    data = {}
    for key in keys:
        if isinstance(key, tuple):
            assert len(key) == 2
            key_src, key_tgt = key[:2]
        else:
            key_src = key_tgt = key
        data[key_tgt] = results[key_src]

    meta = {}
    if len(meta_keys) != 0:
        for key in meta_keys:
            if isinstance(key, tuple):
                assert len(key) == 2
                key_src, key_tgt = key[:2]
            else:
                key_src = key_tgt = key
            meta[key_tgt] = results[key_src]
    if 'bbox_id' in results:
        meta['bbox_id'] = results['bbox_id']
    data[meta_name] = DC(meta, cpu_only=True)

    return data

def preprocess(results):
    results = load_image(results)
    results = top_down_affine(results)
    results = to_tensor(results)
    results = normalize_tensor(results)
    results = collect(results)
    return results