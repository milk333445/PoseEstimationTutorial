# Pose Tools

## poseannotator.py
**Labeling yolov8pose format tool**
```python=
dataset_path = './original'
finish_path = './finish_imgs'
keypoints_yaml = './settings/keypoints.yaml'
annotator = PoseLabelTask(dataset_path, finish_path, keypoints_yaml)
annotator.run()
```
After labeling, you will get the dataset in yolov8pose format.

## transfer_multi2single.py
**If you want to train mmpose, because it's a top-down algorithm, you have to slice and dice the individual objects first**
```python=
dataset_path = './finish_imgs'
keypoints_yaml = './settings/keypoints.yaml'
save_path = './dataset_seperate'
annotator = TransferMulti2Single(dataset_path, keypoints_yaml, save_path)
annotator.run()
```
## yolo2mmposeformat_train_val.py
**After using transfer_multi2single.py to slice the yolov8pose dataset, you can use this tool to convert yolov8pose formate dataset to mmpose format(mscoco).**
```python=
input_dir = './dataset_seperate'
output_dir = './mmpose_format'
converter = YoloToMMPoseConverter(input_dir, output_dir, ratio=0.9)
converter.convert()
```
## mmdetect_results_vidualize.py、mmpose_results_vidualize.py
**These two tools can visualize the log data after mmpose and mmdetection training.**
### mmpose
```python=
log_path = './work_dirs/rtmpose-m-carpose/20240421_225307/vis_data/scalars.json' # 訓練日誌
df_train, df_test = process_log(log_path)
plot_metrics(df_train, df_test, loss_metrics, acc_metrics, ap_metrics, test_metrics, get_line_arg)
```
### mmdetection
```python=
log_path = './work_dirs/faster_r_cnn_triangle/20231228_113814/vis_data/scalars.json' # 訓練日誌
df_train, df_test = process_log(log_path)
plot_metrics(df_train, df_test, loss_metrics, acc_metrics, ap_metrics, test_metrics, get_line_arg)
```

