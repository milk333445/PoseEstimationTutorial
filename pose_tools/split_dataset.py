import os
import shutil
import random


def split_dataset(dataset_dir, train_ratio):
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    
    train_images_dir = os.path.join(images_dir, 'train')
    val_images_dir = os.path.join(images_dir, 'val')
    train_labels_dir = os.path.join(labels_dir, 'train')
    val_labels_dir = os.path.join(labels_dir, 'val')
    
    for directory in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    random.shuffle(images)  
    
    split_index = int(len(images) * train_ratio)
    
    train_images = images[:split_index]
    val_images = images[split_index:]
    
    for image in train_images:
        shutil.move(os.path.join(images_dir, image), os.path.join(train_images_dir, image))
        shutil.move(os.path.join(labels_dir, image.replace('.jpg', '.txt')), os.path.join(train_labels_dir, image.replace('.jpg', '.txt')))
    
    for image in val_images:
        shutil.move(os.path.join(images_dir, image), os.path.join(val_images_dir, image))
        shutil.move(os.path.join(labels_dir, image.replace('.jpg', '.txt')), os.path.join(val_labels_dir, image.replace('.jpg', '.txt')))
    
    print("Dataset split complete.")
    
    
if __name__ == '__main__':
    dataset_directory = './dataset_seperate'
    split_ratio = 0.95
    split_dataset(dataset_directory, split_ratio)