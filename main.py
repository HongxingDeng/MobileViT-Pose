# Copyright (c) OpenMMLab. All rights reserved.
import os
import time

import cv2
import numpy as np
import camera_config_2i as camera_configs
from crestereo import CREStereo

import argparse
import csv
import json
import mimetypes
import os
import time

import cv2
import mmcv
import mmengine
import numpy as np
from mmdeploy_runtime import Detector, PoseDetector
from mmengine.structures import InstanceData

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.registry import VISUALIZERS
from mmpose.structures import split_instances


def DepthMap(imgs, KL, KR, baseline):
    """
        This function is used to computing the depth map.
        Params Description:
            - 'imgs' is the list of the image data.
            - 'K' is the intrinsic matrix of the input images.
            - 'baseline' is the baseline of the stereo camera.
    """
    # Getting the focal length.
    fx = KL[0][0]
    fy = KL[1][1]
    xl = KL[0][2]
    xr = KR[0][2]

    # Initializing the depth map list.
    depthMaps = []
    # Computing the depth maps.
    for i in range(len(imgs)):
        # Getting the image size.
        h = imgs[i].shape[0]
        # Initializing the depth map.
        depthMap = np.zeros(h)
        # Creating the depth map.
        for k in range(0, h):
            b = (imgs[i][k] + abs(xl - xr))
            depthMap[k] = fx * baseline / b
        depthMaps.append(depthMap)
    # Returning the depth map.
    return depthMaps


def process_one_image(args,
                      img,
                      detector,
                      pose_detector,
                      visualizer=None,
                      wait_time=int):
    # apply detector
    bboxes, labels, _ = detector(img)
    # filter detections
    keep = np.logical_and(labels == 0, bboxes[..., 4] > 0.6)
    pre_bbox = bboxes[keep, :5]
    bboxes = bboxes[keep, :4]
    bboxe_scores = pre_bbox[..., 4]
    # apply pose detector
    poses = pose_detector(img, bboxes)
    kpts = visualize(img, poses, bboxes, bboxe_scores, 0.5, (1242, 2208), vis=visualizer)
    # #  show
    img_vis = visualizer.get_image()
    cv2.imshow('images', img_vis)
    cv2.waitKey(1)
    return bboxe_scores, kpts


def visualize(img, keypoints, bboxes, bbox_scores, kpt_thr=0.5, resize=(1242, 2208), vis=None):
    pred_instances = InstanceData()
    width_scale = resize[0] / img.shape[0]
    hight_scale = resize[1] / img.shape[1]
    scores = keypoints[..., 2]
    for i in range(0, bboxes.shape[0]):
        bboxes[i] = [bboxes[i][0] * width_scale, bboxes[i][1] * hight_scale, bboxes[i][2] * width_scale,
                     bboxes[i][3] * hight_scale]
        for j in range(0, keypoints[i].shape[0]):
            keypoints[i][j, 0] = keypoints[i][j, 0] * width_scale
            keypoints[i][j, 1] = keypoints[i][j, 1] * hight_scale
    # vis
    pred_instances.__setattr__('keypoint_scores', scores)
    pred_instances.__setattr__('keypoints', keypoints)
    pred_instances.__setattr__('bboxes', bboxes)
    pred_instances.__setattr__('bbox_scores', bbox_scores)
    img = cv2.resize(img, (0, 0), fx=width_scale, fy=hight_scale)
    pre_img = vis._draw_instances_bbox(img, pred_instances)
    pre_img = vis._draw_instances_kpts(pre_img, pred_instances, kpt_thr=kpt_thr, show_kpt_idx=True)
    return keypoints

def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use SDK Python API')
    parser.add_argument('--device_name', help='name of device, cuda or cpu', default='cuda')
    parser.add_argument(
        '--det_model_path',default='work_dir/yolox',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument(
        '--pose_model_path',default='work_dir/simcc',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('--input', default='datasets/00/0/left', help='path of input image')
    parser.add_argument(
        '--radius',
        type=int,
        default=1,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument('--show', default=True)
    parser.add_argument('--output_root', default='res.jpg')
    parser.add_argument('--save_predictions', default=True)
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--out', default=0, help='Sleep seconds per frame')
    args = parser.parse_args()
    return args





args = parse_args()
# Model Selection options (not all options supported together)
iters = 5  # Lower iterations are faster, but will lower detail.
# Options: 2, 5, 10, 20

shape = (240, 320)  # Input resolution.
# Options: (120,160), (160,240) , (240,320), (360,640), (480,640)

version = "init"  # The combined version does 2 passes, one to get an initial estimation and a second one to refine it.
# Options: "init", "combined"

# Initialize model
model_path = f'/mnt/data1/ONNX-CREStereo-Depth-Estimation-main/models/crestereo_{version}_iter{iters}_{shape[0]}x{shape[1]}.onnx'
depth_estimator = CREStereo(model_path)
# # create object detector
detector = Detector(
    model_path=args.det_model_path, device_name=args.device_name)
# create pose detector
pose_detector = PoseDetector(
    model_path=args.pose_model_path, device_name=args.device_name)


vis = {}
vis_backends = {'type': 'LocalVisBackend'}
vis['type'] = 'PoseLocalVisualizer'
vis['vis_backends'] = vis_backends
vis['name'] = 'visualizer'
vis['radius'] = args.radius
vis['alpha'] = args.alpha
vis['line_width'] = args.thickness
dataset_meta = parse_pose_metainfo(
    dict(from_file='mmpose-dev-1.x/configs/_base_/datasets/coco.py'))
visualizer = VISUALIZERS.build(vis)
visualizer.set_dataset_meta(dataset_meta)
path = '/mnt/data1/mobileViT-Pose/datasets/00/0/'
filelist = os.listdir(path + 'left')
all_time = 0
for file in filelist:
    if file.split('.')[-1] == 'png':
        pass
    else:
        continue
    left_path = path + 'left/' + file
    right_path = path + 'right/right' + file.strip('left')
    csv_path = path + 'result/' + version + str(iters) + '_' + str(shape[0]) + 'x' + str(shape[1]) + '_' + \
               file.split('.')[0] + '.csv'
    out_path = '/mnt/data1/mobileViT-Pose/datasets/00/0/result/depth' + file
    # csv_path =
    # Load images
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    start_time = time.monotonic()
    bbox_score, kpts = process_one_image(args, img=left_img, detector=detector, pose_detector=pose_detector,
                                         visualizer=visualizer)
    keypoint_inf_time = time.monotonic() - start_time

    left_img = cv2.resize(left_img, dsize=(shape[1], shape[0]), fx=1, fy=1)
    right_img = cv2.resize(right_img, dsize=(shape[1], shape[0]), fx=1, fy=1)
    # Estimate the depth
    disparity_map, crestereo_inf_time = depth_estimator(left_img, right_img)
    inf_time = keypoint_inf_time + 0.02101
    all_time += inf_time
    print('keypoint inf time: {}; crestereo inf time: {}; inf_time: {};fps: {}'.format(
        keypoint_inf_time, crestereo_inf_time, inf_time, (1/inf_time)))
    # # disp and csv
    # t = float(2208) / float(shape[1])
    # disparity = cv2.resize(disparity_map, dsize=(2208, 1242), fx=1, fy=1, interpolation=cv2.INTER_LINEAR) * t
    # depthMaps = DepthMap(disparity, camera_configs.left_camera_matrix, camera_configs.right_camera_matrix,
    #                      camera_configs.t)
    # depth = np.array(depthMaps, dtype=float)
    # np.savetxt(csv_path, depth, delimiter=",")
print(10/all_time)