import os
import shutil
import cv2



def test_images_replicate(folder1, folder2):
    if not os.path.exists(folder1) or not os.path.exists(folder2):
        print("Folder does not exist")
    sames_files = []
    for file in os.listdir(folder1):
        filename = os.fsdecode(file)
        # check if file is in folder2
        if not os.path.exists(os.path.join(folder2, filename)):
            pass
            
        else:
            sames_files.append(filename)
    print(sames_files)
    print(len(sames_files))

folder1 = './dataset/images/train'
folder2 = 'new_dataset/imgs'
labels_folder1 = './dataset/labels/train'
labels_folder2 = './new_labels'
test_images_replicate(folder1, folder2)
test_images_replicate(labels_folder1, labels_folder2)