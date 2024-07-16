# <div align="center"> YOLO PoseEstimator </div>

## A. Introduction
The YOLO PoseEstimator Package provides a versatile and user-friendly platform for pose estimation, supporting YOLO models in formats PyTorch. With its **PoseEstimatorBuilder** class, users can easily manage multiple models through a YAML configuration, offering the flexibility to run pose estimation globally or on individual models to meet diverse needs.

## B. Feature
- Customizable pose estimation settings such as confidence thresholds and IOU thresholds.
- Easy management of multiple models with automatic handling of device allocation (CPU/GPU).

## C. Prerequisites
- Python 3.8 or higher
- OpenCV
- PyTorch (with CUDA support)
- torchvision
- NumPy
- yaml
- ultralytics
- CUDA and cuDNN (for GPU support)

## D. Usage
### 1. Set Up Configuration
Create a YAML configuration file specifying the model paths, sizes, thresholds, and other settings. Example structure for config.yml:
#### Model Configuration
```YAML
model_A:
    weight: "./weights/yolov8x-pose.pt"
    draw_settings: './configuration/body_pose_settings.yaml'
    size: 640
    conf_threshold: 0.5
    iou_threshold: 0.5
    fp16: false
    verbose: false
    classes:
model_B:
    weight: "./weights/yolov8x-pose.pt"
    draw_settings: './configuration/body_pose_settings.yaml'
    size: 640
    conf_threshold: 0.5
    iou_threshold: 0.5
    fp16: false
    verbose: false
    classes:
...
```
#### Draw Configuration
Because pose visualization requires more configurations, you need to write a yaml to control it.
```YAML
bbox_color: [150, 0, 0]
bbox_thickness: 3

bbox_labelstr:
  font_size: 1
  font_thickness: 2
  offset_x: 0
  offset_y: -15

kpt_color_map:
  0:
    name: Nose
    color: [0, 0, 255]
    radius: 6
  1:
    name: Right Eye
    color: [255, 0, 0]
    radius: 6
  2:
    name: Left Eye
    color: [255, 0, 0]
    radius: 6
  ...

kpt_labelstr:
  font_size: 0.5
  font_thickness: 1
  offset_x: 10
  offset_y: 0

skeleton_map:
  - srt_kpt_id: 15
    dst_kpt_id: 13
    color: [0, 100, 255]
    thickness: 2
  - srt_kpt_id: 13
    dst_kpt_id: 11
    color: [0, 255, 0]
    thickness: 2
  - srt_kpt_id: 16
    dst_kpt_id: 14
    color: [255, 0, 0]
    thickness: 2
  ...
```

&nbsp;

### 2. Load and Run
Use the PoseEstimatorBuilder class to load the models specified in your configuration file.  Then, run pose estimation on your images as shown in the following example:
```python
from nexva.pose import PoseEstimatorBuilder

# Initialize the detector
detector = PoseEstimatorBuilder('config.yml')

# Sample image loading (add your own image loading mechanism)
images = [cv2.imread('path/to/image.jpg')]

# Run all pose estimation models
detector.run(images)

# Or execute pose estimation using only the 'model_A' specified by its key in the YAML configuration
results = detector.model_A.run(images)

# visualization 
for i, img in enumerate(images):
    img = detector.model_A.draw_pose_results_on_images(img, results[i]['bbox'], results[i]['keypoints'], show_id=False, conf_thres=0.5)
```
**API reference:**
- **PoseEstimatorBuilder(file)**: Class to manage the loading and running of specified pose estimation models based on a YAML configuration file.
- **run(images)**: Method to process a list of images through the pose estimation models.
    + images: list of image nparrays.

# <div align="center"> MM PoseEstimator </div>
## A. Introduction
The MM PoseEstimator Package provides a versatile and user-friendly platform for pose estimation, supporting models in formats PyTorch. With its **MMposeEstimatorBuilder** class, users can easily manage multiple models through a YAML configuration, offering the flexibility to run pose estimation globally or on individual models to meet diverse needs.

## B. Feature
- Customizable pose estimation settings such as confidence thresholds and IOU thresholds.
- Easy management of multiple models with automatic handling of device allocation (CPU/GPU).

## C. Prerequisites
- Python 3.8 or higher
- OpenCV
- PyTorch (with CUDA support)
- torchvision
- NumPy
- yaml
- openmim
- mmengine
- mmcv>=2.0.1
- mmdet>=3.1.0
- mmpose>=1.1.0
- CUDA and cuDNN (for GPU support)
- mmpose repo(https://github.com/open-mmlab/mmpose)
- mmdetection repo(https://github.com/open-mmlab/mmdetection)

## D. Usage
### 1. Set Up Configuration
Create a YAML configuration file specifying the model paths, sizes, thresholds, and other settings. Example structure for config.yml:
#### Model Configuration
```YAML
model_A:
  draw_setting_path: './settings/draw_settings.yaml'
  detector:
    config_path: 'mmpose/projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py'
    checkpoint_path: 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'

  pose:
    config_path: 'mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py'
    checkpoint_path: 'mmpose/checkpoint/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'

  conf_thresh: 0.5
  iou_thresh: 0.3
...
```
#### Draw Configuration
Because pose visualization requires more configurations, you need to write a yaml to control it.
```YAML
bbox_color: [150, 0, 0]
bbox_thickness: 3

bbox_labelstr:
  font_size: 1
  font_thickness: 2
  offset_x: 0
  offset_y: -15

kpt_color_map:
  0:
    name: Nose
    color: [0, 0, 255]
    radius: 6
  1:
    name: Right Eye
    color: [255, 0, 0]
    radius: 6
  2:
    name: Left Eye
    color: [255, 0, 0]
    radius: 6
  ...

kpt_labelstr:
  font_size: 0.5
  font_thickness: 1
  offset_x: 10
  offset_y: 0

skeleton_map:
  - srt_kpt_id: 15
    dst_kpt_id: 13
    color: [0, 100, 255]
    thickness: 2
  - srt_kpt_id: 13
    dst_kpt_id: 11
    color: [0, 255, 0]
    thickness: 2
  - srt_kpt_id: 16
    dst_kpt_id: 14
    color: [255, 0, 0]
    thickness: 2
  ...
```

&nbsp;

### 2. Load and Run
Use the MMposeEstimatorBuilder class to load the models specified in your configuration file.  Then, run pose estimation on your images as shown in the following example:

**Remember add MMposeEstimatorBuilder to the init file.**
```python
path:./nexva/nexva/pose/__init__.py
from .mmpose_builder import MMposeEstimatorBuilder
```

```python
from nexva.pose import MMposeEstimatorBuilder

# Initialize the detector
detector = MMposeEstimatorBuilder('config.yml')

# Sample image loading (add your own image loading mechanism)
images = [cv2.imread('path/to/image.jpg')]

# Run all pose estimation models
detector.run(images)

# Or execute pose estimation using only the 'model_A' specified by its key in the YAML configuration
results = detector.model_A.run(images)

# visualization 
for i, img in enumerate(images):
    img = detector.model_A.draw_pose_results_on_images(img, results[i]['bbox'], results[i]['keypoints'], show_id=False, conf_thres=0.5)
```
**API reference:**
- **MMposeEstimatorBuilder(file)**: Class to manage the loading and running of specified pose estimation models based on a YAML configuration file.
- **run(images)**: Method to process a list of images through the pose estimation models.
    + images: list of image nparrays.