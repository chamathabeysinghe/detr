import cv2
import os
import numpy as np


def write_file(vid_frames, file_name):
    height, width, layers = vid_frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(file_name, fourcc, 6.0, (width, height))

    for i in range(len(vid_frames)):
        out.write(vid_frames[i].astype(np.uint8))
    out.release()


IMG_PATH = '/Users/cabe0006/Projects/monash/Datasets/eval_output'
OUTPUT_PATH = '/Users/cabe0006/Projects/monash/Datasets/vid_output'
track_first_n_frames = 5
vid_indexes = [2, 3]

for v in vid_indexes:
    print('Video index: {}'.format(v))
    vid_frames = []
    for i in range(track_first_n_frames):
        img_id = v * track_first_n_frames + i
        img = cv2.imread(os.path.join(IMG_PATH, '{}.jpg'.format(img_id)))
        vid_frames.append(img)
    write_file(vid_frames, os.path.join(OUTPUT_PATH, '{}.mp4'.format(v)))












