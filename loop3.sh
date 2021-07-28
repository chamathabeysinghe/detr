#!/bin/bash
#for i in CU10L1B1In_0 CU10L1B1Out_0 CU10L1B4In_0 CU10L1B4Out_0 CU10L1B5In_0 CU10L1B5Out_0 CU15L1B1In_0 CU15L1B1Out_0 CU15L1B4In_0 CU15L1B4Out_0 OU10B1L1In_0 OU10B1L1Out_0 OU10B1L2In_0 OU10B1L2Out_0 OU10B1L3In_0 OU10B1L3Out_0 OU10B2L1In_0 OU10B2L1Out_0 OU10B2L2In_0 OU10B2L2Out_0 OU10B2L3In_0 OU10B2L3Out_0 OU10B3L1In_0 OU10B3L1Out_0 OU10B3L2In_0 OU10B3L2Out_0 OU10B3L3In_0 OU10B3L3Out_0
for i in CU20L1B1In_0 CU20L1B1Out_0 CU20L1B4In_0 CU20L1B4Out_0 CU25L1B1In_0 CU25L1B1Out_0 CU25L1B4In_0 CU25L1B4Out_0 OU50B1L1In_0 OU50B1L1Out_0 OU50B1L2In_0 OU50B1L2Out_0 OU50B1L3In_0 OU50B1L3Out_0
do
  /home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/data/ant_dataset_images/$i --output_dir /home/cabe0006/mb20_scratch/chamath/data/ant_dataset_medium_image_predictions/$i --resume /home/cabe0006/mb20_scratch/chamath/detr-v4/checkpoints/checkpoint.pth --thresh 0.75

#  python test.py --value=$i
#  python -c 'print "a"*'$i
done