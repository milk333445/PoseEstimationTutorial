
import os
import sys
import yaml
import torch
from ultralytics.nn.autobackend import AutoBackend
from .yolopose_estimator import PoseEstimator


class PoseEstimatorBuilder():
    def __init__(self, file):
        assert file.endswith(('.yaml', '.yml'))
        with open(file, 'r') as f:
            setting = yaml.safe_load(f)
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for key in setting.keys():
            draw_settings = setting[key]['draw_settings']
            model = AutoBackend(
                weights=setting[key]['weight'],
                device=device,
                fp16=setting[key]['fp16'],
                verbose=setting[key]['verbose']
                
            )
            pose_estimator = PoseEstimator(
                model,
                device,
                setting[key]['size'], 
                draw_settings,
                setting[key]['conf_threshold'], 
                setting[key]['iou_threshold'],
                setting[key]['fp16'],
                setting[key]['classes']
                )
            setattr(self, key, pose_estimator)
            setattr(self, key + '_cls', setting[key]['classes'])