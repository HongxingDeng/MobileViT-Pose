# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from typing import Sequence

import cv2
import mmcv
import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmengine import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar

from mmpose.apis import init_model, inference_topdown
from mmpose.evaluation import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize feature map')
    parser.add_argument(
        '--img', default='../2.jpg')
    parser.add_argument('--det_config',
                        default='../demo/mmdetection_cfg/yolox_tiny_8xb8-300e_coco.py',
                        help='Config file for detection')
    parser.add_argument('--det_checkpoint',
                        default='../epoch_95.pth',
                        help='Checkpoint file for detection')

    parser.add_argument('--pose_config',
                        default='../work_dirs/rtmpose-mobilevit-xx-small-DWS-SG-CoordConv-old/featmap_rtmpose-mobilevit-xx-small-DWS-SG-CoordConv.py')
    parser.add_argument('--pose_checkpoint', default='../work_dirs/rtmpose-mobilevit-xx-small-DWS-SG-CoordConv-old/best_coco_AP_epoch_415.pth')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--target-layers',
        default=['backbone'],
        nargs='+',
        type=str,
        help='The target layers to get feature map, if not set, the tool will '
        'specify the backbone')
    parser.add_argument(
        '--preview-model',
        default=False,
        action='store_true',
        help='To preview all the model layers')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--show', action='store_true', help='Show the featmap results')
    parser.add_argument(
        '--channel-reduction',
        default='squeeze_mean',
        help='Reduce multiple channels to a single channel')
    parser.add_argument(
        '--topk',
        type=int,
        default=4,
        help='Select topk channel to show by the sum of each channel')
    parser.add_argument(
        '--arrangement',
        nargs='+',
        type=int,
        default=[2, 2],
        help='The arrangement of featmap when channel_reduction is '
        'not None and topk > 0')
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
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


class ActivationsWrapper:

    def __init__(self, detector, model, target_layers):
        self.detector = detector
        self.model = model
        self.activations = []
        self.handles = []
        self.image = None
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def __call__(self, img_path):
        args = parse_args()
        self.activations = []
        det_result = inference_detector(self.detector, img_path)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                       pred_instance.scores > args.bbox_thr)]
        bboxes = bboxes[nms(bboxes, args.nms_thr), :4]
        # pose_results = inference_topdown(self.model, img_path, bboxes)
        # results = inference_detector(self.model, img_path)
        pose_results = inference_topdown(self.model, img_path)
        return pose_results, self.activations

    def release(self):
        for handle in self.handles:
            handle.remove()


def init_pose_estimator(pose_config, pose_checkpoint, device, cfg_options):
    pass


def main():
    args = parse_args()
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    cfg = Config.fromfile(args.pose_config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # bbox det
    init_default_scope(cfg.get('default_scope', 'mmyolo'))
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    channel_reduction = args.channel_reduction
    if channel_reduction == 'None':
        channel_reduction = None
    assert len(args.arrangement) == 2

    # build pose estimator
    model = init_model(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))

    if not os.path.exists(args.out_dir) and not args.show:
        os.mkdir(args.out_dir)

    if args.preview_model:
        print(model)
        print('\n This flag is only show model, if you want to continue, '
              'please remove `--preview-model` to get the feature map.')
        return

    target_layers = []
    for target_layer in args.target_layers:
        try:
            target_layers.append(eval(f'model.{target_layer}'))
        except Exception as e:
            print(model)
            raise RuntimeError('layer does not exist', e)

    activations_wrapper = ActivationsWrapper(detector, model, target_layers)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta



    result, featmaps = activations_wrapper(args.img)
    if not isinstance(featmaps, Sequence):
        featmaps = [featmaps]

    flatten_featmaps = []
    for featmap in featmaps:
        if isinstance(featmap, Sequence):
            flatten_featmaps.extend(featmap)
        else:
            flatten_featmaps.append(featmap)

    img = mmcv.imread(args.img)
    img = mmcv.imconvert(img, 'bgr', 'rgb')

    # show the results
    shown_imgs = []
    # visualizer.add_datasample(
    #     'result',
    #     img,
    #     data_sample=result,
    #     draw_gt=False,
    #     show=False,
    #     wait_time=0,
    #     out_file=None,
    #     pred_score_thr=args.score_thr)
    data_samples = merge_data_samples(result)
    visualizer.add_datasample(
        'result',
        img,
        data_sample=data_samples,
        draw_gt=False,
        draw_heatmap=False,
        draw_bbox=True,
        show_kpt_idx=False,
        skeleton_style=True,
        show=False,
        wait_time=100,
        kpt_thr=args.kpt_thr)
    drawn_img = visualizer.get_image()
    img = cv2.resize(img, dsize=(256, 192), fx=1, fy=1)
    for featmap in flatten_featmaps:
        shown_img = visualizer.draw_featmap(
            featmap[0],
            img,
            resize_shape=(192,256),
            channel_reduction=channel_reduction,
            topk=args.topk,
            arrangement=args.arrangement)
        shown_imgs.append(shown_img)
    mmcv.imwrite(mmcv.rgb2bgr(shown_imgs[0]), '1.jpg')


# Please refer to the usage tutorial:
# https://github.com/open-mmlab/mmyolo/blob/main/docs/zh_cn/user_guides/visualization.md # noqa
if __name__ == '__main__':
    main()
