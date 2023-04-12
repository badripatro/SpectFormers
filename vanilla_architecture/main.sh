#!/bin/bash

echo "Current path is $PATH"
echo "Running"
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# CUDA_LAUNCH_BLOCKING=1 
python -m torch.distributed.launch --nproc_per_node=8 --use_env  main_spectformer.py  --output_dir logs/spectformer-xs --arch spectformer-xs --batch-size 128 --data-path ../../../dataset/Image_net/imagenet/ --data-set IMNET  --num_workers 12 --epochs 320

# Transfer Learning 
# python -m torch.distributed.launch --nproc_per_node=8 --use_env main_spectformer_transfer.py  --output_dir logs/spectformer-b-cifar10 --arch spectformer-b --batch-size 64 --data-set CIFAR10 --data-path ../../../dataset/Image_net/ --epochs 1000 --lr 0.0001 --weight-decay 1e-4 --clip-grad 1 --warmup-epochs 5 --finetune ../spectformer_mvt_b/logs/spectformer-b/checkpoint_best.pth 
# python -m torch.distributed.launch --nproc_per_node=8 --use_env main_spectformer_transfer.py  --output_dir logs/spectformer-b-cifar100 --arch spectformer-b --batch-size 64 --data-set CIFAR100 --data-path ../../../dataset/Image_net/ --epochs 1000 --lr 0.0001 --weight-decay 1e-4 --clip-grad 1 --warmup-epochs 5 --finetune ../spectformer_mvt_b/logs/spectformer-b/checkpoint_best.pth 
# python -m torch.distributed.launch --nproc_per_node=8 --use_env main_spectformer_transfer.py  --output_dir logs/spectformer-b-FLOWERS --arch spectformer-b --batch-size 64 --data-set FLOWERS --data-path ../../../dataset/Image_net/flowers --epochs 1000 --lr 0.0001 --weight-decay 1e-4 --clip-grad 1 --warmup-epochs 5 --finetune ../spectformer_mvt_b/logs/spectformer-b/checkpoint_best.pth 
# python -m torch.distributed.launch --nproc_per_node=8 --use_env main_spectformer_transfer.py  --output_dir logs/spectformer-b-pet --arch spectformer-b --batch-size 64 --data-set PET --data-path ../../../dataset/Image_net/pets --epochs 1000 --lr 0.0001 --weight-decay 1e-4 --clip-grad 1 --warmup-epochs 5 --finetune ../spectformer_mvt_b/logs/spectformer-b/checkpoint_best.pth 
# python -m torch.distributed.launch --nproc_per_node=8 --use_env main_spectformer_transfer.py  --output_dir logs/spectformer-b-car --arch spectformer-b --batch-size 64 --data-set CARS --data-path ../../../dataset/Image_net/cars --epochs 1000 --lr 0.0001 --weight-decay 1e-4 --clip-grad 1 --warmup-epochs 5 --finetune ../spectformer_mvt_b/logs/spectformer-b/checkpoint_best.pth 
