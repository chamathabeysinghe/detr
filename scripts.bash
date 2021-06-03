## Local Development Environment

python main.py --device cpu --num_workers 0 --dataset_file ant --data_path /Users/cabe0006/Projects/monash/Datasets/dataset-small/ --output_dir /Users/cabe0006/Projects/monash/detr/output/output_5 --resume /Users/cabe0006/Projects/monash/detr/checkpoints/detr-r50-e632da11.pth
python main.py --device cpu --num_workers 0 --dataset_file ant --data_path /Users/cabe0006/Projects/monash/Datasets/dataset-small/ --output_dir /Users/cabe0006/Projects/monash/detr/output/output_5 --resume /Users/cabe0006/Projects/monash/detr/output/output_5/checkpoint.pth --eval

python test.py --device cpu --dataset_file ant --data_path /Users/cabe0006/Projects/monash/Datasets/eval --output_dir /Users/cabe0006/Projects/monash/detr/output/output_3 --resume /Users/cabe0006/Projects/monash/detr/output/output_3/checkpoint.pth

python visualizer.py --device cpu --dataset_file ant --data_path /Users/cabe0006/Projects/monash/Datasets/eval --output_dir /Users/cabe0006/Projects/monash/Datasets/eval_output2 --resume /Users/cabe0006/Projects/monash/detr/output/output_4/checkpoint.pth



## Server Environment

# Start training
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/detr/dataset/ --output_dir /home/cabe0006/mb20_scratch/chamath/detr/output_backbone_freeze --resume /home/cabe0006/mb20_scratch/chamath/detr/output_backbone_freeze/checkpoint.pth
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/detr/dataset/ --output_dir /home/cabe0006/mb20_scratch/chamath/detr/output_backbone_freeze --eval --resume /home/cabe0006/mb20_scratch/chamath/detr/output_backbone_freeze/checkpoint.pth

# Resume training
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/detr/dataset/ --output_dir /home/cabe0006/mb20_scratch/chamath/detr/output --resume /home/cabe0006/mb20_scratch/chamath/detr/output/checkpoint.pth
# Draw box images
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/detr/dataset/train/ --output_dir /home/cabe0006/mb20_scratch/chamath/detr/eval_output_frozen_backbone/train --resume /home/cabe0006/mb20_scratch/chamath/detr/output/checkpoint.pth
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/detr/dataset/test/ --output_dir /home/cabe0006/mb20_scratch/chamath/detr/eval_output_with_backbone/test --resume /home/cabe0006/mb20_scratch/chamath/detr/output/checkpoint.pth

