#!/bin/bash
for i in CU10L1B1In_0, CU10L1B1Out_0
do
  /home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/data/ant_dataset_images/$i --output_dir /home/cabe0006/mb20_scratch/chamath/data/ant_dataset_image_predictions/$i --resume /home/cabe0006/mb20_scratch/chamath/detr-v4/checkpoints/checkpoint.pth --thresh 0.75

#  python test.py --value=$i
#  python -c 'print "a"*'$i
done