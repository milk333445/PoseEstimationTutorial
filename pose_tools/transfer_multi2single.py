import yaml
import os
import cv2

class TransferMulti2Single:
    def __init__(self, dataset_path, keypoints_path, save_path):
        self.dataset_path = dataset_path
        self.keypoints = self.load_keypoints(keypoints_path)
        self.images = self.load_images(dataset_path)
        self.img_idx = 0
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.img_save_path = os.path.join(self.save_path, 'images')
        self.label_save_path = os.path.join(self.save_path, 'labels')
        if not os.path.exists(self.img_save_path):
            os.makedirs(self.img_save_path)
        if not os.path.exists(self.label_save_path):
            os.makedirs(self.label_save_path)

    def load_images(self, dataset_path):
        imgs_path = f'{dataset_path}/images'
        images = [os.path.join(imgs_path, f) for f in os.listdir(imgs_path) if f.endswith('.jpg')]
        return images
    
    def load_keypoints(self, keypoints_yaml):
        with open(keypoints_yaml, 'r') as file:
            keypoints_config = yaml.safe_load(file)
        return keypoints_config
    
    def yolo_pose_2_keypoints(self, pose_file):
        with open(pose_file, 'r') as file:
            lines = file.readlines()
            bboxes = []
            kpts = []
            for line in lines:
                line_data = [float(x) for x in line.strip().split()]
                cx, cy, delta_x, delta_y = line_data[1:5]
                x0, x1, y0, y1 = (cx - delta_x / 2) * self.img_w, (cx + delta_x / 2) * self.img_w, (cy - delta_y / 2) * self.img_h, (cy + delta_y / 2) * self.img_h
                bboxes.append([x0, x1, y0, y1])
                point_coordinates = [(line_data[i] * self.img_w, line_data[i+1] * self.img_h) for i in range(5, 29, 3)]
                l_f_b, l_r_b, l_f_t, l_r_t, r_f_b, r_r_b, r_f_t, r_r_t = point_coordinates
                kpts.append([l_f_b, l_r_b, l_f_t, l_r_t, r_f_b, r_r_b, r_f_t, r_r_t])
        return bboxes, kpts

    def save_single_bbox_to_yolo_format(self, cropped_img, keypoints, i, idx):
        normalized_bbox = [0.5, 0.5, 1, 1]
        normalized_keypoints = [(kpt[0] / cropped_img.shape[1], kpt[1] / cropped_img.shape[0]) for kpt in keypoints]
        
        # skip if at least 6 keypoints are missing(0, 0, 0)
        if sum([1 for kpt in normalized_keypoints if kpt[0] == 0 and kpt[1] == 0]) >= 6:
            print(f'Skipped image {i}_{idx} due to missing keypoints')
            return

        img_save_path = os.path.join(self.img_save_path, f'{i}_{idx}.jpg')
        cv2.imwrite(img_save_path, cropped_img)
        print(f'Saved image to {img_save_path}')
        label_save_path = os.path.join(self.label_save_path, f'{i}_{idx}.txt')
        with open(label_save_path, 'w') as file:
            file.write(f'0 {normalized_bbox[0]} {normalized_bbox[1]} {normalized_bbox[2]} {normalized_bbox[3]}')
            for kpt in normalized_keypoints:
                if kpt[0] != 0 and kpt[1] != 0:
                    file.write(f' {kpt[0]} {kpt[1]} 2')
                else:
                    file.write(f' 0 0 0')
            print(f'Saved label to {label_save_path}')
            print('-----------------------------------')

    def run(self):
        for i, img_path in enumerate(self.images):
            labels_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
            img = cv2.imread(img_path)
            self.img_h, self.img_w = img.shape[:2]
            bboxes, keypoints = self.yolo_pose_2_keypoints(labels_path)

            for idx, (box, kpts) in enumerate(zip(bboxes, keypoints)):
                tmp_img = img.copy()
                cropped_img = tmp_img[int(box[2]):int(box[3]), int(box[0]):int(box[1])]
                cropped_keypoints = []
                for kpt in kpts:
                    if kpt[0] != 0 and kpt[1] != 0:
                        cropped_keypoints.append((kpt[0] - box[0], kpt[1] - box[2]))
                    else:
                        cropped_keypoints.append((0, 0))
                # save single bbox to yolo format
                self.save_single_bbox_to_yolo_format(cropped_img, cropped_keypoints, i, idx)
       

if __name__ == '__main__':
    dataset_path = './dataset_copy'
    keypoints_yaml = './settings/keypoints.yaml'
    save_path = './dataset_seperate'
    annotator = TransferMulti2Single(dataset_path, keypoints_yaml, save_path)
    annotator.run()