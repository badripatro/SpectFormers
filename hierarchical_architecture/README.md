# SpectFormer Hierarchical Model

### Requirement:
* PyTorch 1.10.0+
* Python3.8
* CUDA 10.1+
* [timm](https://github.com/rwightman/pytorch-image-models)==0.4.5
* [tlt](https://github.com/zihangJiang/TokenLabeling)==0.1.0
* pyyaml
* apex-amp


## Data Preparation

Download and extract ImageNet images from http://image-net.org/. The directory structure should be

```

│ILSVRC2012/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......

```


### Model Zoo

We provide baseline SVT Hierarchical models pre-trained on ImageNet1k 2012, using the distilled version of our method:

| name | resolution | #params | FLOPs | Top-1 Acc. | Top-5 Acc. |
| :---: | :---: | :---: | :---: | :---: | :---: |
| SpectFormer-H-S | 224 | 22.2M | 3.9 | 84.3  | 96.9 | 
| SpectFormer-H-B | 224 | 33.1M | 6.3 |  85.0 | 97.1 | 
| SpectFormer-H-L | 224 | 54.7M | 12.7 | 85.7 | 97.3 | 


### Train SpectFormer small model
```
python3 -m torch.distributed.launch \
   --nproc_per_node=8 \
   --nnodes=1 \
   --node_rank=0 \
   --master_addr="localhost" \
   --master_port=12346 \
   --use_env main.py --config configs/spectformer/spectformer_s.py --data-path /export/home/dataset/imagenet --epochs 310 --batch-size 128 \
   --token-label --token-label-size 7 --token-label-data /export/home/dataset/imagenet/label_top5_train_nfnet
```


### Train SpectFormer Base model
```
python3 -m torch.distributed.launch \
   --nproc_per_node=8 \
   --nnodes=1 \
   --node_rank=0 \
   --master_addr="localhost" \
   --master_port=12346 \
   --use_env main.py --config configs/spectformer/spectformer_b.py --data-path /export/home/dataset/imagenet --epochs 310 --batch-size 128 \
   --token-label --token-label-size 7 --token-label-data /export/home/dataset/imagenet/label_top5_train_nfnet
```



### Train SpectFormer Large model
```
python3 -m torch.distributed.launch \
   --nproc_per_node=8 \
   --nnodes=1 \
   --node_rank=0 \
   --master_addr="localhost" \
   --master_port=12346 \
   --use_env main.py --config configs/spectformer/spectformer_l.py --data-path /export/home/dataset/imagenet --epochs 310 --batch-size 128 \
   --token-label --token-label-size 7 --token-label-data /export/home/dataset/imagenet/label_top5_train_nfnet
```


## Citation

If you find this repo helpful, please consider citing us.

```
@article{patro2023spectformer,
  title={SpectFormer: Frequency and Attention is what you need in a Vision Transformer},
  author={Patro, Badri N and Namboodiri, Vinay P and Agneeswaran, Vijay Srinivas},
  journal={arXiv preprint arXiv:2304.06446},
  year={2023}
}

```


# Acknowledgements
Our code is based on [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [DeiT](https://github.com/facebookresearch/deit), [WaveVit](https://github.com/YehLi/ImageNetModel)  and [GFNet](https://github.com/raoyongming/GFNet).
