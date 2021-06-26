import cv2
import numpy as np
import os

BASE_DIR = '/home/cabe0006/mb20_scratch/chamath/data'
PREDICTION_DIR = os.path.join(BASE_DIR, 'ant_dataset_detr_predictions', 'images')
VIDEO_DIR = os.path.join(BASE_DIR, 'ant_dataset_detr_predictions', 'videos')
os.makedirs(VIDEO_DIR, exist_ok=True)

TAGGED = 'tagged'

IN_DIR = os.path.join(PREDICTION_DIR, TAGGED)
OUT_DIR = os.path.join(VIDEO_DIR, TAGGED)
os.makedirs(OUT_DIR, exist_ok=True)


def read_write_dir(dir, out_file):
    print(out_file)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    files = sorted(os.listdir(dir))
    out = None
    for i, file in enumerate(files):
        img = cv2.imread(os.path.join(dir, file))
        if i == 0:
            height, width, layers = img[0].shape
            out = cv2.VideoWriter(out_file, fourcc, 6.0, (width, height))

        out.write(img.astype(np.uint8))
    out.release()


img_dirs = os.listdir(IN_DIR)
for img_dir in img_dirs:
    output_file = os.path.join(OUT_DIR, f"{img_dir}.mp4")
    in_dir_path = os.path.join(IN_DIR, img_dir)
    read_write_dir(in_dir_path, output_file)

