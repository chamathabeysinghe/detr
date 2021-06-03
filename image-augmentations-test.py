from datasets import build_dataset, get_coco_api_from_dataset
import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import util.misc as utils
import matplotlib.pyplot as plt


class Args:
    def __init__(self):
        self.dataset_file = 'ant2'
        self.data_path = '/Users/cabe0006/Projects/monash/Datasets/dataset-small/'
        self.masks = False


args = Args()
train_dataset = build_dataset(image_set='train', args=args)
validation_dataset = build_dataset(image_set='val', args=args)

img = train_dataset[0][0].permute(1, 2, 0).numpy()
plt.imshow(img)
plt.show()
