import json
import os
import cv2
import random
import copy


class YoloToMMPoseConverter:
    def __init__(self, input_dir, output_dir, ratio=0.9):
        self.input_images_dir = os.path.join(input_dir, 'images')
        self.input_labels_dir = os.path.join(input_dir, 'labels')
        self.output_dir = output_dir
        self.ratio = ratio
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.new_images_dir = os.path.join(self.output_dir, 'images')
        if not os.path.exists(self.new_images_dir):
            os.makedirs(self.new_images_dir)
        self.initialize_data()
        self.train_img_idx = 0
        self.val_img_idx = 0
        self.train_counter = 0
        self.val_counter = 0
        
      
    def initialize_data(self):
        self.base_structure = {
            "categories": [
                {
                    "supercategory": "car",
                    "id": 1,
                    "name": "car",
                    "keypoints": [
                        'left-front-bumper',
                        'left-rear-bumper',
                        'left-front-tire',
                        'left-rear-tire',
                        'right-front-bumper',
                        'right-rear-bumper',
                        'right-front-tire',
                        'right-rear-tire'
                    ],
                    "skeleton": [[0, 1], [2, 3], [4, 5], [6, 7]]
                }
            ],
            "images": [],
            "annotations": []
        }
        self.train_data = copy.deepcopy(self.base_structure)
        self.val_data = copy.deepcopy(self.base_structure)
        
    def convert(self):
        print("Converting dataset to mmpose format...")
        self.process_images()
        print("Conversion complete.")
        self.save_json()
        print("Json file saved.")
        
    def process_images(self):
        images = [f for f in os.listdir(self.input_images_dir) if f.endswith('.jpg')]
        random.shuffle(images)
        total_images = len(images)
        split_idx = int(total_images * self.ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        for img_set, dataset, type in zip([train_images, val_images], [self.train_data, self.val_data], ['train', 'val']):
            for idx, img_file in enumerate(img_set):
                print(f"Processing image {idx + 1}/{len(img_set)}: {img_file}")
                img_path = os.path.join(self.input_images_dir, img_file)
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
                dataset['images'].append(
                    {
                        "file_name": img_file,
                        "height": h,
                        "width": w,
                        "id": self.train_img_idx if type == 'train' else self.val_img_idx
                    }
                )
                
                if type == 'train':
                    self.process_labels(self.train_img_idx, img_file, h, w, dataset, type)
                    self.train_img_idx += 1
                else:
                    self.process_labels(self.val_img_idx, img_file, h, w, dataset, type)
                    self.val_img_idx += 1
                
                # save image
                new_img_path = os.path.join(self.new_images_dir, img_file)
                cv2.imwrite(new_img_path, img)
                print(f"Saved image to {new_img_path}")
                   
    def process_labels(self, img_idx, img_file, h, w, dataset, type):
        label_file = os.path.join(self.input_labels_dir, img_file.replace('.jpg', '.txt'))
        with open(label_file, 'r') as file:
            print(f'Processing labels: {label_file}')
            for line in file:
                parts = line.split()
                keypoints = []
                # cx, cy, w, h
                cx = float(parts[1]) * w
                cy = float(parts[2]) * h
                delta_x = float(parts[3]) * w
                delta_y = float(parts[4]) * h
                x0 = int(cx - delta_x / 2)
                x1 = int(cx + delta_x / 2)
                y0 = int(cy - delta_y / 2)
                y1 = int(cy + delta_y / 2)
                bbox_w = x1 - x0
                bbox_h = y1 - y0
                bbox = [x0, y0, bbox_w, bbox_h]
                for i in range(5, len(parts), 3):
                    x = float(parts[i]) * w
                    y = float(parts[i + 1]) * h
                    v = int(float(parts[i + 2]))
                    keypoints.extend([int(x), int(y), v])
                
                dataset['annotations'].append(
                    {
                       'category_id': 1,
                       'segmentation': [],
                       "iscrowd": 0,
                       "image_id": img_idx,
                       'id': self.train_counter if type == 'train' else self.val_counter,
                       'bbox': bbox,
                       'area': delta_x * delta_y,
                       'num_keypoints': len(keypoints) // 3,
                       'keypoints': keypoints
                    }
                )
                
                if dataset == self.train_data:
                    self.train_counter += 1
                else:
                    self.val_counter += 1
                
                    
    def save_json(self):
        train_output = os.path.join(self.output_dir, 'train_coco.json')
        val_output = os.path.join(self.output_dir, 'val_coco.json')
        try:
            with open(train_output, 'w') as json_file:
                json.dump(self.train_data, json_file, indent=4)
                print(f"Train JSON file saved to {train_output}")
            with open(val_output, 'w') as json_file:
                json.dump(self.val_data, json_file, indent=4)
                print(f"Validation JSON file saved to {val_output}")
        except Exception as e:
            print("Failed to save JSON:", str(e))
            print("Checking train_data structure:")
    
if __name__ == '__main__':
    input_dir = './dataset_copy'
    output_dir = './mmpose_format'
    converter = YoloToMMPoseConverter(input_dir, output_dir, ratio=0.9)
    converter.convert()