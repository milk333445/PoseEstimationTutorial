import os
import json
import shutil
import numpy as np
from tqdm import tqdm

# 框的類別
bbox_class = {
   'car': 0,
}
keypoint_class = [
    'left_front_bumper',
    'left_rear_bumper',
    'right_front_bumper',
    'right_rear_bumper'
]

def process_single_json(labelme_path, save_folder):
    
    with open(labelme_path, 'r', encoding='utf-8') as f:
        labelme = json.load(f)
    img_width = labelme['imageWidth']   
    img_height = labelme['imageHeight'] 
    
    suffix = labelme_path.split('.')[-2]
    yolo_txt_path = suffix + '.txt'
    processed_keypoints = set() # 處理標記過的關鍵點
    
    with open(yolo_txt_path, 'w', encoding='utf-8') as f:

        for each_ann in labelme['shapes']: 

            if each_ann['shape_type'] == 'rectangle': 

                yolo_str = ''

                bbox_class_id = bbox_class[each_ann['label']]
                yolo_str += '{} '.format(bbox_class_id)
               
                # 以左上角和右下角的座標表示
                bbox_top_left_x = int(min(each_ann['points'][0][0], each_ann['points'][1][0]))
                bbox_bottom_right_x = int(max(each_ann['points'][0][0], each_ann['points'][1][0]))
                bbox_top_left_y = int(min(each_ann['points'][0][1], each_ann['points'][1][1]))
                bbox_bottom_right_y = int(max(each_ann['points'][0][1], each_ann['points'][1][1]))
                # 以中心點和寬高表示
                bbox_center_x = int((bbox_top_left_x + bbox_bottom_right_x) / 2)
                bbox_center_y = int((bbox_top_left_y + bbox_bottom_right_y) / 2)
                # 框的寬高
                bbox_width = bbox_bottom_right_x - bbox_top_left_x
                bbox_height = bbox_bottom_right_y - bbox_top_left_y
                # 歸一化
                bbox_center_x_norm = bbox_center_x / img_width
                bbox_center_y_norm = bbox_center_y / img_height
                bbox_width_norm = bbox_width / img_width
                bbox_height_norm = bbox_height / img_height
                yolo_str += '{:.5f} {:.5f} {:.5f} {:.5f} '.format(bbox_center_x_norm, bbox_center_y_norm, bbox_width_norm, bbox_height_norm)

             
                bbox_keypoints_dict = {}
                for each_ann in labelme['shapes']: 
                    if each_ann['shape_type'] == 'point': 
                 
                        x = int(each_ann['points'][0][0])
                        y = int(each_ann['points'][0][1])
                        label = each_ann['label']
                        visibility = 2
                        point_id = (x, y)
                        if point_id not in processed_keypoints and (x>bbox_top_left_x) & (x<bbox_bottom_right_x) & (y<bbox_bottom_right_y) & (y>bbox_top_left_y): # 筛选出在该个体框中的关键点
                            bbox_keypoints_dict[label] = [x, y, visibility]
                            processed_keypoints.add(point_id)

                for each_class in keypoint_class: 
                    if each_class in bbox_keypoints_dict:
                        keypoint_x_norm = bbox_keypoints_dict[each_class][0] / img_width
                        keypoint_y_norm = bbox_keypoints_dict[each_class][1] / img_height
                        visibility = bbox_keypoints_dict[each_class][2]
                        yolo_str += '{:.5f} {:.5f} {} '.format(keypoint_x_norm, keypoint_y_norm, visibility) # 2-可见不遮挡 1-遮挡 0-没有点
                    else: 
                        yolo_str += '0 0 0 '
                f.write(yolo_str + '\n')
    shutil.move(yolo_txt_path, save_folder)
    print('{} --> {} 轉換完成'.format(labelme_path, yolo_txt_path))
    
def process_all_json(labelme_folder, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for each_json in tqdm(os.listdir(labelme_folder)):
        if each_json.endswith('.json'):
            labelme_path = os.path.join(labelme_folder, each_json)
            labelme_path = os.path.normpath(labelme_path)
            try:
                process_single_json(labelme_path, save_folder)
            except:
                print('{} 轉換失敗'.format(labelme_path))
    print('所有json文件轉換完成')
    
if __name__ == '__main__':
    labelme_folder = './labels'
    save_folder = './new_labels'
    process_all_json(labelme_folder, save_folder)