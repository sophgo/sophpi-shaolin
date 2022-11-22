import bmnett

bmnett.compile(model="../mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb",
               outdir="./Mask_RCNN_bmodel",
               descs="[0, uint8, 0, 256]",
               shapes=[1, 1920, 1920, 3],
               dyn=False,
               net_name="maskrcnn",
               input_names=["image_tensor"],
               output_names=["detection_boxes", "detection_classes", "detection_masks", "detection_scores",
                             "num_detections"],
               opt=2,
               target="BM1684"
               )
