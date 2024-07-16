import cv2
import os
import yaml
import numpy as np
import copy

from nexva.vision import color_list

class StatusWindow:
    def __init__(self, window_name, pos):
        self.pos = pos
        self.status_window_name = window_name
        self.status_text = 'Init'
        cv2.namedWindow(self.status_window_name)
        cv2.moveWindow(self.status_window_name, pos[0], pos[1])

        self.update_status(self.status_text)

    def update_status(self, status):
        self.status_text = status
        cv2.imshow(self.status_window_name, self.get_status_image())
        cv2.moveWindow(self.status_window_name, self.pos[0], self.pos[1])
        cv2.setWindowProperty(self.status_window_name, cv2.WND_PROP_TOPMOST, 1)

    def get_status_image(self):
        status_img = np.ones((100, 1000, 3), dtype=np.uint8) * 255  # 空白背景
        text_size, _ = cv2.getTextSize(self.status_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = (status_img.shape[1] - text_size[0]) // 2  
        text_y = (status_img.shape[0] + text_size[1]) // 2  
        cv2.putText(status_img, self.status_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return status_img

class PoseAnnotator:

    def __init__(self, keypoints):
        self.results = []
        self.state = 'init'
        self.keypoints = keypoints
        cv2.namedWindow('image')
        self.status_window = StatusWindow('status', (500, 800))

    def run(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (1280, 720))
        self.orig_img = img
        self.prev_img = copy.deepcopy(img)
        self.base_img = copy.deepcopy(img)
        self.curr_img = copy.deepcopy(img)
        self.img_h, self.img_w = self.curr_img.shape[:2]
        while True:
            if self.state == 'init':
                self.state = self.draw_bbox
            elif self.state == 'break':
                self.save_results()
                break
            elif self.state == 'quit':
                self.quit_flag = True
                return False
            else:
                self.state()
        cv2.destroyAllWindows()
        return self.save_results()

    def draw_bbox(self):
        cv2.setMouseCallback('image', self._mouse_callback_bbox)
        self.bbox_state = 'init'
        self.status_window.update_status('Drawing Bounding Box')

        while True:

            cv2.imshow('image', self.curr_img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                self.state = 'quit'
                break

            elif key == 13:
                self.state = 'break'
                break

            elif key == ord('b') :
                copy_orig_img = copy.deepcopy(self.orig_img)
                self.curr_img = self.draw_last_results(copy_orig_img, self.results)
                self.prev_img = copy.deepcopy(self.curr_img)

            elif self.bbox_state == 'done': #enter
                self.state = self.draw_keypoints
                break

    def _mouse_callback_bbox(self, event, x, y, flags, param):

        if event == cv2.EVENT_MOUSEMOVE :
            self.curr_img = copy.deepcopy(self.prev_img)
            cv2.line(self.curr_img, (x, 0), (x, self.img_h), (0, 0, 0), 2)
            cv2.line(self.curr_img, (0, y), (self.img_w, y), (0, 0, 0), 2)
            if flags == cv2.EVENT_FLAG_LBUTTON:
                self.curr_img = copy.deepcopy(self.base_img)
                cv2.rectangle(self.curr_img, self.tmp_bboxes[:2], (x, y), (0, 255, 0), 2)
                self.prev_img = copy.deepcopy(self.curr_img)
        
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.tmp_bboxes = [x,y,None,None]

        elif event == cv2.EVENT_LBUTTONUP:
            self.tmp_bboxes[2:4] = [x, y]
            if self.tmp_bboxes[2] < 0:
                self.tmp_bboxes[2] = 0
            elif self.tmp_bboxes[2] >= self.img_w:
                self.tmp_bboxes[2] = self.img_w - 1
            if self.tmp_bboxes[3] < 0:  
                self.tmp_bboxes[3] = 0
            elif self.tmp_bboxes[3] >= self.img_h:
                self.tmp_bboxes[3] = self.img_h - 1
            
            self.curr_img = copy.deepcopy(self.base_img)
            cv2.rectangle(self.curr_img, self.tmp_bboxes[:2], self.tmp_bboxes[2:], (0, 255, 0), 2)
            self.prev_img = copy.deepcopy(self.curr_img)
            self.status_window.update_status("'Right click': continue, 'Enter': finish ")
                
        elif event == cv2.EVENT_RBUTTONDOWN and self.tmp_bboxes[-1] is not None:
            self.results.append({'bbox': [self.tmp_bboxes]})
            self.curr_img = copy.deepcopy(self.base_img)
            self.curr_img = self.draw_last_results(self.curr_img, self.results)
            self.prev_img = copy.deepcopy(self.curr_img)
            self.base_img = copy.deepcopy(self.curr_img)
            self.bbox_state = 'done'

    def draw_keypoints(self):
        self.kpts = [[]]
        self.results[-1]['keypoints'] = []
        cv2.setMouseCallback('image', self._mouse_callback_keypoints)
        self.kpts_state = 'init'
        self.status_window.update_status(f"Drawing Keypoint: {self.keypoints[0]['name']}")

        while True:
            cv2.imshow('image', self.curr_img)
            if self.kpts:
                if len(self.kpts[-1]) >= len(self.keypoints):
                    self.status_window.update_status(f"'Right click': continue, 'ENTER': finish, 'b': back")
                else:
                    self.status_window.update_status(f"Keypoint: {self.keypoints[(len(self.kpts[-1]))]['name']}, 'SPACE': cannot find, 'b': back")

            key = cv2.waitKey(1) & 0xFF

            if self.kpts_state == 'draw':
                if key == ord('w'):
                    self.kpts[-1][-1][1] -= 1
                elif key == ord('s'):
                    self.kpts[-1][-1][1] += 1
                elif key == ord('a'):
                    self.kpts[-1][-1][0] -= 1
                    
                elif key == ord('d'):
                    self.kpts[-1][-1][0] += 1

                if key in [ord('w'), ord('s'), ord('a'), ord('d')]:
                    print(self.results[-1]['keypoints'])
                    self.results[-1].update({'keypoints': self.kpts[-1]})
                    print(self.results[-1]['keypoints'])
                    copy_orig_img = copy.deepcopy(self.orig_img)
                    self.curr_img = self.draw_last_results(copy_orig_img, self.results)
                    self.prev_img = copy.deepcopy(self.curr_img)

            if key == 27:
                self.state = 'quit'
                self.results = []
                break

            elif key == 32: #space
                if len(self.kpts[-1]) < len(self.keypoints):
                    self.kpts[-1].append([0, 0, 0])
                    self.results[-1].update({'keypoints': self.kpts[-1]})
                continue
            elif key == ord('b') :
                while True:
                    try:
                        ret = self.results[-1]['keypoints'].pop()
                        if ret:
                            break
                    except IndexError:
                        break
                copy_orig_img = copy.deepcopy(self.orig_img)
                self.curr_img = self.draw_last_results(copy_orig_img, self.results)
                self.prev_img = copy.deepcopy(self.curr_img)

            elif key == 13: # enter
                if len(self.kpts[-1]) == len(self.keypoints):
                    self.state = 'break'
                    break
                else:
                    continue

            elif self.kpts_state == 'done':
                self.state = self.draw_bbox
                break

    def _mouse_callback_keypoints(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.kpts_state == 'init':
                self.kpts_state = 'draw'

            if len(self.kpts[-1]) < len(self.keypoints):
                if x < self.results[-1]['bbox'][0][0] or x > self.results[-1]['bbox'][0][2] or y < self.results[-1]['bbox'][0][1] or y > self.results[-1]['bbox'][0][3]:
                    return
                self.kpts[-1].append([x, y, 2])
                self.results[-1].update({'keypoints': self.kpts[-1]})
                color = color_list[len(self.kpts[-1]) - 1]
                cv2.circle(self.curr_img, (x, y), 4, color, cv2.FILLED)
                cv2.putText(self.curr_img, str(len(self.kpts[-1]) - 1), (int(x - 10), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

                self.prev_img = copy.deepcopy(self.curr_img)

            elif len(self.kpts[-1]) == len(self.keypoints):
                self.status_window.update_status(f"Keypoints Done: press Right click to continue")

        elif event == cv2.EVENT_RBUTTONDOWN and len(self.kpts[-1]) == len(self.keypoints):
            self.prev_img = copy.deepcopy(self.curr_img)
            self.base_img = copy.deepcopy(self.curr_img)
            self.kpts_state = 'done'
            self.results[-1].update({'keypoints': self.kpts[-1]})
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if len(self.kpts[-1]) < len(self.keypoints):
                self.curr_img = copy.deepcopy(self.prev_img)
                cv2.circle(self.curr_img, (x, y), 4, color_list[len(self.kpts[-1])], -1)  # Green circle
                cv2.putText(self.curr_img, str(len(self.kpts[-1])), (int(x - 10), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


    @staticmethod
    def draw_last_results(img, results):
        for obj in results:
            if 'bbox' in obj:
                for bbox in obj['bbox']:
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            if 'keypoints' in obj:
                for i, kpt in enumerate(obj['keypoints']):
                    if kpt:
                        cv2.circle(img, (kpt[0], kpt[1]), 4, color_list[i], cv2.FILLED)
                        cv2.putText(img, str(i), (int(kpt[0] - 10), int(kpt[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


        return img

    def save_results(self):
        if len(self.results) == 0:
            return False
        norm_results = []
        for results in self.results:
            norm_results.append({'bbox': None, 'keypoints': None})
            for key, value in results.items():
                if key == 'bbox':
                    normalized_bbox = np.array(value) / [self.img_w, self.img_h, self.img_w, self.img_h]
                    normalized_bbox_rounded = np.round(normalized_bbox, 6)
                    norm_results[-1]['bbox'] =  normalized_bbox_rounded.tolist()
                elif key == 'keypoints':
                    normalized_keypoints = np.array(value) / [self.img_w, self.img_h, 1]
                    normalized_keypoints_rounded = np.round(normalized_keypoints, 6)
                    norm_results[-1]['keypoints'] =  normalized_keypoints_rounded.tolist()

        return norm_results


class PoseLabelTask:
    def __init__(self, dataset_path, finish_path, keypoints_yaml):
        self.dataset_path = dataset_path
        self.finish_imgs_path = f'{finish_path}/images'
        self.finish_labels_path = f'{finish_path}/labels'
        os.makedirs(self.finish_imgs_path, exist_ok=True)
        os.makedirs(self.finish_labels_path, exist_ok=True)
        self.keypoints = self.load_keypoints(keypoints_yaml)['keypoints']
        self.images = self.load_images(dataset_path)
        self.curr_img = None
        self.img_index = 0
            
    def load_keypoints(self, keypoints_yaml):
        with open(keypoints_yaml, 'r') as file:
            keypoints_config = yaml.safe_load(file)
        return keypoints_config
    
    def load_images(self, dataset_path):
        images = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]
        images.sort()
        return images
    
    def run(self):
        pose_label = PoseAnnotator(self.keypoints)
        for img_path in self.images:
            print(f'now is processing {img_path}')
            results = pose_label.run(img_path)
            if results:
                file_name = self.finish_labels_path + '/' + img_path.split('/')[-1].split('.jpg')[0]
                new_img_path = self.finish_imgs_path + '/' + img_path.split('/')[-1] 
                print(f'mv {img_path} {new_img_path}')
                os.system(f'mv {img_path} {new_img_path}')
                self.write_label_file(file_name, results)
            if pose_label.quit_flag == True:
                break      

    @staticmethod
    def write_label_file(file_name, results):
        with open(file_name + '.txt', 'w') as f:
            for result in results:
                bbox = result['bbox'][0]
                x0 = min(bbox[0], bbox[2])
                x1 = max(bbox[0], bbox[2])
                y0 = min(bbox[1], bbox[3])
                y1 = max(bbox[1], bbox[3])

                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                w = x1 - x0
                h = y1 - y0

                f.write(f'0 {cx} {cy} {w} {h} ')
                for keypoint in result['keypoints']:
                    f.write(f'{keypoint[0]} {keypoint[1]} {keypoint[2]} ')
                f.write('\n')

if __name__ == '__main__':
    dataset_path = './original'
    finish_path = './finish_imgs'
    keypoints_yaml = './settings/keypoints.yaml'
    annotator = PoseLabelTask(dataset_path, finish_path, keypoints_yaml)
    annotator.run()