# Copyright (c) OpenMMLab. All rights reserved.
import csv
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    print(pose_results)
    # print(pose_results)
    data_samples = merge_data_samples(pose_results)
    # print(data_samples.get('pred_instances')['keypoints'], None)
    pre_kpts = np.array(data_samples.get('pred_instances')['keypoints'])
    kpt_score = np.array(data_samples.get('pred_instances')['keypoint_scores'])

    # print(kpt_score)
    with open('test.csv', 'a+', newline='') as f:
        write = csv.writer(f)
        for i in range(0, pre_kpts.shape[0]):
            kpts = pre_kpts[i]
            score = kpt_score[i]
            kpts_with_score = np.insert(kpts, 2, score, axis=1)
            for kpt_write in kpts_with_score:
                write.writerow(kpt_write)
        f.close()

    # with open()
    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='../data/coco/val2017/', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='SIGMA',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
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
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')
    parser.add_argument(
        '--save_predictions', action='store_true', help='Draw bboxes of instances')
    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None
    path = args.input
    filelist = os.listdir(path)
    for image in filelist:
        if image.split('.')[-1] != 'jpg' and image.split('.')[-1] != 'png':
            print(image.split('.')[-1])
            continue
        image_path = path +image
        output_file = None
        if args.output_root:
            mmengine.mkdir_or_exist(args.output_root)
            output_file = os.path.join(args.output_root,
                                       os.path.basename(image_path))

        if True:
            # assert args.output_root != ''
            args.pred_save_path = f'{args.output_root}/' \
                f'{os.path.splitext(os.path.basename(image_path))[0]}.json'

        # build detector

        detector = init_detector(
            args.det_config, args.det_checkpoint, device=args.device)
        detector.cfg = adapt_mmdet_pipeline(detector.cfg)

        # build pose estimator
        pose_estimator = init_pose_estimator(
            args.pose_config,
            args.pose_checkpoint,
            device=args.device,
            cfg_options=dict(
                model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

        # build visualizer
        pose_estimator.cfg.visualizer.radius = args.radius
        pose_estimator.cfg.visualizer.alpha = args.alpha
        pose_estimator.cfg.visualizer.line_width = args.thickness
        visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
        # the dataset_meta is loaded from the checkpoint and
        # then pass to the model in init_pose_estimator
        visualizer.set_dataset_meta(
            pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)
        input_type = 'image'

        if input_type == 'image':

            # inference
            pred_instances = process_one_image(args, image_path, detector,
                                               pose_estimator, visualizer)

            if args.save_predictions:
                pred_instances_list = split_instances(pred_instances)

            if output_file:
                img_vis = visualizer.get_image()
                mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

        else:
            args.save_predictions = False
            raise ValueError(
                f'file {os.path.basename(image_path)} has invalid format.')

        if args.save_predictions:
            with open(args.pred_save_path, 'w') as f:
                json.dump(
                    dict(
                        meta_info=pose_estimator.dataset_meta,
                        instance_info=pred_instances_list),
                    f,
                    indent='\t')
            print(f'predictions have been saved at {args.pred_save_path}')

if __name__ == '__main__':
    main()
