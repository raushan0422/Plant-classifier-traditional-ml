import pandas as pd
import os
import cv2

def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    return df

def load_images(image_dir, image_ids, img_size=(128, 128)):
    images = []
    for image_id in image_ids:
        img_path = os.path.join(image_dir, f"{image_id}.jpg")
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        images.append(img)
    return images
