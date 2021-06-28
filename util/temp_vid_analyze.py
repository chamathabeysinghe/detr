import cv2
import os
import pandas as pd

base_dir = '/Users/cabe0006/Projects/monash/Datasets/ant_dataset_small'
tagged = 'tagged'
filtered = 'all'
vid_dir = os.path.join(base_dir, tagged)

vid_files = list(filter(lambda x: 'mp4' in x, os.listdir(vid_dir)))
if filtered == 'only_one':
    vid_files = list(filter(lambda x: '_0' in x, vid_files))


def get_file_details(file, file_path):
    cap = cv2.VideoCapture(file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    duration = float(frame_count) / float(fps)
    detection_count = int(file[2:4]) * frame_count
    parent_folder = file.split('_')[0]
    if ('In' in file):
        detection_count = detection_count / 2
    return [file, parent_folder, frame_count, fps, duration, detection_count]


records = {}
for i, file in enumerate(vid_files):
    file_path = os.path.join(vid_dir, file)
    record = get_file_details(file, file_path)
    parent_folder = record[1]
    if parent_folder not in records:
        records[parent_folder] = [record]
    else:
        records[parent_folder].append(record)

summary_records = []
for key in records:
    #     file, parent_folder, frame_count, fps, duration, detection_count
    items = records[key]
    frame_count = 0
    duration = 0.0
    detection_count = 0.0
    fps = items[0][3]
    for item in items:
        frame_count += item[2]
        duration += item[4]
        detection_count += item[5]
    summary_records.append([key, int(fps), frame_count, duration, detection_count])

df = pd.DataFrame(summary_records, columns =['parent_folder', 'fps', 'frmae_count', 'duration', 'detection_count'])
df.to_csv(os.path.join(base_dir, f'{tagged}_{filtered}.csv'))

