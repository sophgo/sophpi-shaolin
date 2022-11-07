import os
import warnings
from argparse import ArgumentParser


from inference import inference_top_down_pose_model

from utils import init_model, COCO
from draw import vis_pose_result

def main():
    """Visualize the demo images.

    Require the json_file containing boxes.
    """
    parser = ArgumentParser()
    parser.add_argument('--bmodel-path', type=str, default='../dataset/mmpose_fp32.bmodel', help='bmodel path')
    parser.add_argument('--img-root', type=str, default='../dataset/', help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default='../dataset/test_coco.json',
        help='Json file containing image info.')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='vis_results',
        help='Root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--tpu_id',
        type=int,
        default=0,
        help='tpu id for inference')

    args = parser.parse_args()

    coco = COCO(args.json_file)

    img_keys = list(coco.imgs.keys())

    # optional
    return_heatmap = False

    output_layer_names = None
    bmodel = init_model(args.bmodel_path, tpu_id=args.tpu_id)
    # process each image
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

        # test a single image, with a list of bboxes
        pose_results, returned_outputs = inference_top_down_pose_model(
            bmodel,
            image_name,
            person_results,
            bbox_thr=None,
            format='xywh',
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(args.out_img_root, f'vis_{i}.jpg')

        vis_pose_result(
            image_name,
            pose_results,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            out_file=out_file)


if __name__ == '__main__':
    main()
