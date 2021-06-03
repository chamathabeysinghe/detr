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

# /home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/detr/dataset/test/ --output_dir /home/cabe0006/mb20_scratch/chamath/detr/eval_output/test --resume /home/cabe0006/mb20_scratch/chamath/detr/output/checkpoint.pth

IMG_PATH = '/home/cabe0006/mb20_scratch/chamath/detr/eval_output_with_backbone/test'
OUTPUT_PATH = '/home/cabe0006/mb20_scratch/chamath/detr/video_output_with_backbone'
track_first_n_frames = 500
vid_indexes = [2, 3]

os.makedirs(OUTPUT_PATH, exist_ok=True)

for v in vid_indexes:
    print('Video index: {}'.format(v))
    vid_frames = []
    for i in range(track_first_n_frames):
        print(i)
        img_id = v * track_first_n_frames + i
        img = cv2.imread(os.path.join(IMG_PATH, '{}.jpg'.format(img_id)))
        print(os.path.join(IMG_PATH, '{}.jpg'.format(img_id)))
        vid_frames.append(img)
    write_file(vid_frames, os.path.join(OUTPUT_PATH, '{}.mp4'.format(v)))












