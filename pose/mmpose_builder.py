import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import mmcv
import numpy as np
import time
import yaml
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmdet.apis import inference_detector, init_detector
from .mmpose_estimator import MMposeEstimator



class MMposeEstimatorBuilder:
    def __init__(self, file):
        assert file.endswith(('.yaml', '.yml'))
        with open(file, 'r') as f:
            setting = yaml.safe_load(f)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for key in setting.keys():
            detector_config = setting[key]['detector']['config_path']
            detector_checkpoint = setting[key]['detector']['checkpoint_path']
            pose_config = setting[key]['pose']['config_path']
            pose_checkpoint = setting[key]['pose']['checkpoint_path']
            draw_setting_path = setting[key]['draw_setting_path']
            conf_thresh = setting[key]['conf_thresh']
            iou_thresh = setting[key]['iou_thresh']

            detector = init_detector(
                detector_config,
                detector_checkpoint,
                device=device
            )
            pose_estimator = init_pose_estimator(
                pose_config,
                pose_checkpoint,
                device=device,
                cfg_options={'model': {'test_cfg': {'output_heatmaps': False}}}
            )

            mmpose_estimator = MMposeEstimator(detector, pose_estimator, draw_setting_path, conf_thresh, iou_thresh, device)
            setattr(self, key, mmpose_estimator)