Run command

python -m torch.distributed.run --master_port 29522  main_resnet.py --model kernresnet50  --kern 7  --batch_size 512 --data-path /scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization  --output_dir ./output --mixup 0 --cutmix 0 --smoothing 0  --color-jitter 0 --num_workers 8 --patch_dataset  --k_patch 10 --lr 0.01 
