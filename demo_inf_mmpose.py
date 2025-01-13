from argparse import ArgumentParser

import cv2
import mmcv
import numpy as np

from mmdeploy.apis import inference_model
from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.registry import VISUALIZERS

parser = ArgumentParser()
parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
parser.add_argument(
    '--thickness',
    type=int,
    default=1,
    help='Link thickness for visualization')
parser.add_argument(
    '--alpha', type=float, default=0.8, help='The transparency of bboxes')


args = parser.parse_args()
img = '2.jpg'
result = inference_model(
  model_cfg='mmpose/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w32_dark-8xb64-210e_coco'
            '-wholebody-256x192.py',
  deploy_cfg='mmdeploy/configs/mmpose/pose-detection_tensorrt-fp16_static-256x192.py',
  backend_files=['work_dir/whol/end2end.engine'],
  img=img,
  device='cuda:0')
kpts = result[0].get('pred_instances').get('keypoints')

img = mmcv.imread(img)
print(result)
# build visualizer
vis = {}
vis_backends = {'type': 'LocalVisBackend'}
vis['type']='PoseLocalVisualizer'
vis['vis_backends'] = vis_backends
vis['name'] = 'visualizer'
vis['radius'] = args.radius
vis['alpha'] = args.alpha
vis['line_width'] = args.thickness
visualizer = VISUALIZERS.build(vis)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_pose_estimator
dataset_meta = parse_pose_metainfo(
            dict(from_file='mmpose/configs/_base_/datasets/coco_wholebody.py'))
visualizer.set_dataset_meta(
    dataset_meta, skeleton_style='mmpose')

# show the results
if isinstance(img, str):
    img = mmcv.imread(img, channel_order='rgb')
elif isinstance(img, np.ndarray):
    img = mmcv.bgr2rgb(img)

if visualizer is not None:
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result[0],
        draw_gt=False,
        draw_bbox=True,
        show_kpt_idx=False,
        skeleton_style='mmpose',
        show=True,
        wait_time=0,
        kpt_thr=0.3)


# for keypoint in kpts[0]:
#   print(keypoint)
#   cv2.circle(img, (int(keypoint[0]), int(keypoint[1])), 1,
#              (0, 255, 0), -1)
# cv2.imwrite('pose.png', img)