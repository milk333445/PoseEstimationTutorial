import yaml
import os
import cv2

class ShowPoseTask():
    def __init__(self, dataset_path, keypoints_path):
        self.dataset_path = dataset_path
        self.keypoints = self.load_keypoints(keypoints_path)
        self.images = self.load_images(dataset_path)
        self.img_idx = 0

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



    def run(self):
        for i, img_path in enumerate(self.images):
            labels_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
            img = cv2.imread(img_path)
            self.img_h, self.img_w = img.shape[:2]
            bboxes, keypoints = self.yolo_pose_2_keypoints(labels_path)
            for box in bboxes:
                cv2.rectangle(img, (int(box[0]), int(box[2])), (int(box[1]), int(box[3])), (0, 255, 0), 2)
            for kpt in keypoints:
                for point in kpt:
                    cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
            cv2.imshow('img', img)
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()
                continue
        cv2.destroyAllWindows()

if __name__ == '__main__':
    dataset_path = './dataset_seperate'
    keypoints_yaml = './settings/keypoints.yaml'
    annotator = ShowPoseTask(dataset_path, keypoints_yaml)
    annotator.run()