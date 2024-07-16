import os
import json
import numpy as np

def process_single_json(labelme, image_id=1):
    
    global ANN_ID
    
    coco_annotations = []
    
    for each_ann in labelme['shapes']:
        
        if each_ann['shape_type'] == 'rectangle': 
            
            bbox_dict = {}
            bbox_dict['category_id'] = 1
            bbox_dict['segmentation'] = []
            bbox_dict['iscrowd'] = 0
            bbox_dict['image_id'] = image_id
            bbox_dict['id'] = ANN_ID
            
            ANN_ID += 1
            
            bbox_left_top_x = min(int(each_ann['points'][0][0]), int(each_ann['points'][1][0]))
            bbox_left_top_y = min(int(each_ann['points'][0][1]), int(each_ann['points'][1][1]))
            bbox_right_bottom_x = max(int(each_ann['points'][0][0]), int(each_ann['points'][1][0]))
            bbox_right_bottom_y = max(int(each_ann['points'][0][1]), int(each_ann['points'][1][1]))
            bbox_w = bbox_right_bottom_x - bbox_left_top_x
            bbox_h = bbox_right_bottom_y - bbox_left_top_y
            bbox_dict['bbox'] = [bbox_left_top_x, bbox_left_top_y, bbox_w, bbox_h] # 左上角x、y、框的w、h
            bbox_dict['area'] = bbox_w * bbox_h
            
            for each_ann in labelme['shapes']: 
                if each_ann['shape_type'] == 'polygon': 
                    first_x = each_ann['points'][0][0]
                    first_y = each_ann['points'][0][1]
                    if (first_x>bbox_left_top_x) & (first_x<bbox_right_bottom_x) & (first_y<bbox_right_bottom_y) & (first_y>bbox_left_top_y): # 筛选出在该个体框中的关键点
                        bbox_dict['segmentation'] = list(map(lambda x: list(map(lambda y: round(y, 2), x)), each_ann['points']))
            
            # 篩選出個體框中的關鍵點
            bbox_keypoints_dict = {}
            for each_ann in labelme['shapes']:
                
                if each_ann['shape_type'] == 'point':
                    x = int(each_ann['points'][0][0])
                    y = int(each_ann['points'][0][1])
                    label = each_ann['label']
                    if (x>bbox_left_top_x) & (x<bbox_right_bottom_x) & (y<bbox_right_bottom_y) & (y>bbox_left_top_y): # 筛选出在该个体框中的关键点
                        bbox_keypoints_dict[label] = [x, y]
            bbox_dict['num_keypoints'] = len(bbox_keypoints_dict)
            
            # 關鍵點的坐標
            bbox_dict['keypoints'] = []
            for each_class in class_list['keypoints']:
                if each_class in bbox_keypoints_dict:
                    bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][0])
                    bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][1])
                    bbox_dict['keypoints'].append(2) 
                else: 
                    bbox_dict['keypoints'].append(0)
                    bbox_dict['keypoints'].append(0)
                    bbox_dict['keypoints'].append(0)
                    
            coco_annotations.append(bbox_dict)
    return coco_annotations


def process_folder(labels_folder_path, class_list):
    IMG_ID = 0
    ANN_ID = 0
    
    labels_files = os.listdir(labels_folder_path)
    
    coco = {}
    coco['categories'] = []
    coco['categories'].append(class_list)

    coco['images'] = []
    coco['annotations'] = []
    
    for labelme_json in labels_files:
        
        if labelme_json.split('.')[-1] == 'json':
            
            with open(os.path.join(labels_folder_path, labelme_json), 'r', encoding='utf-8') as f:
                labelme = json.load(f)
                
                img_dict = {}
                img_dict['file_name'] = labelme['imagePath']
                img_dict['height'] = labelme['imageHeight']
                img_dict['width'] = labelme['imageWidth']
                img_dict['id'] = IMG_ID
                coco['images'].append(img_dict)
                
                coco_annotations = process_single_json(labelme, IMG_ID)
                coco['annotations'] += coco_annotations
                
                IMG_ID += 1
                
                print(labelme_json, 'is done!')
        else:
            pass
        
        
        
if __name__ == '__main__':
    class_list = {
        'supercategory': 'person',
        'id': 1,
        'name': 'person',
        'keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'],
        'skeleton': [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
    }
    
    labels_folder_path = './labels/train'
    process_folder(labels_folder_path, class_list)