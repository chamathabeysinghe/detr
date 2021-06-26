import os
from multiprocessing import Pool

# video_path = '/home/cabe0006/mb20_scratch/chamath/data/raw_videos'
# dest_dir = '/home/cabe0006/mb20_scratch/chamath/data/frames'


# BASE_DIR = '/Users/cabe0006/Projects/monash/Datasets'
# DATASET = 'ant_dataset_small'

BASE_DIR = '/home/cabe0006/mb20_scratch/chamath/data'
DATASET = 'ant_dataset'
DATASET_DIR = os.path.join(BASE_DIR, DATASET)

DATASET_OUT = 'ant_dataset_images'
DATASET_OUT_DIR = os.path.join(BASE_DIR, DATASET_OUT)

TAGGED_DIR = os.path.join(DATASET_OUT_DIR, 'tagged')
UNTAGGED_DIR = os.path.join(DATASET_OUT_DIR, 'untagged')

os.makedirs(TAGGED_DIR, exist_ok=True)
os.makedirs(UNTAGGED_DIR, exist_ok=True)


video_files = []    ##video_path, tagged, file_name

for t in ['tagged', 'untagged']:
    t_path = os.path.join(DATASET_DIR, t)
    for file in os.listdir(t_path):
        vid_path = os.path.join(t_path, file)
        video_files.append((vid_path, t, file.split('.')[0]))


def convert_frames(vid_file, file_name, dest_dir):
    import cv2
    capture = cv2.VideoCapture(vid_file)
    read_count = 0
    print("Converting video file: {}".format(vid_file))
    while True:
        success, image = capture.read()
        if not success:
            break
        read_count += 1
        path = os.path.join(dest_dir, f"{file_name}_{read_count:06d}.jpg")
        cv2.imwrite(path, image)
        if read_count % 20 == 0:
            print(read_count)


def process_train_vid_file(video_file):
    video_path, tagged, file_name = video_file
    if tagged == 'tagged':
        dest_dir = os.path.join(TAGGED_DIR, file_name)
    else:
        dest_dir = os.path.join(UNTAGGED_DIR, file_name)
    os.makedirs(dest_dir, exist_ok=True)
    convert_frames(video_path, file_name, dest_dir)


with Pool(6) as p:
    p.map(process_train_vid_file, video_files)

