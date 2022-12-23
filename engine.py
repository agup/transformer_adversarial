# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import numpy as np
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import time
from losses import DistillationLoss
import utils
from einops import rearrange

#k_patch = 10
#arg_bs is the args.batch_size
def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, arg_bs,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None, k_patch = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    j = 0
    print("Number of patches k = ", str(k_patch))
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):


        #
        #print('start data time ', time.time())
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        b_size = samples.shape[0]

        if args.patch_dataset:
            #split samples into patches & randomly permute
            
            patches = rearrange(samples, 'b  c (h1 h) (w1 w)  -> (b h1 w1) c h w ', h1=14, w1=14)#.to(device, non_blocking=True)
            idx = torch.randperm(patches.shape[0])
            rand_patches = patches[idx].view(patches.size())#.to(device, non_blocking=True)
            
            #first half of these are label 0
            targets[0:targets.shape[0]//2] =0
            targets[targets.shape[0]//2:]=1
            
            mask_zeros = torch.zeros(b_size, 14*14)
            mask_ones = torch.multinomial(torch.ones(b_size , 14*14 ), k_patch) ##hardocded k = 10 here
            index_x = np.repeat(np.arange(0, b_size ,1),k_patch) ##and here 
            mask_ones = rearrange(mask_ones, 'b h  -> (b  h)')
            mask_zeros[index_x, mask_ones] = 1
            mask_zeros = rearrange(mask_zeros, 'b h  -> (b  h)')
            
            mask = torch.unsqueeze(mask_zeros,1)
            mask = torch.unsqueeze(mask,1)
            mask = torch.unsqueeze(mask,1)
            mask =  mask.to(device, non_blocking=True)

            #print('mask shape ', mask.shape, ' patches shape ', patches.shape, ' rand patches shape ', rand_patches.shape)
            image_ones = mask*(patches) +(1-mask)*rand_patches
            half_patches = patches.shape[0] //2
            samples = torch.cat((rand_patches[0:half_patches], image_ones[half_patches:]  ),0)
            samples =  rearrange(samples, '(b h1 w1) c h w  -> b c (h1 h) (w1 w) ', h1=14, w1=14)
          
            #if j == 0:
            #    np.save('output/sample_images.npy', samples.cpu().detach().numpy())
            #print('samples shape', samples.shape, 'targets shape', targets.shape)
        j += 1
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
        #print('end data time ', time.time())
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        #print("end update " , time.time())
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args = None,  k_patch = None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
    
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        b_size = images.shape[0]
        if args.patch_dataset:
            patches = rearrange(images, 'b  c (h1 h) (w1 w)  -> (b h1 w1) c h w ', h1=14, w1=14)
            idx = torch.randperm(patches.shape[0])
            rand_patches = patches[idx].view(patches.size())

            
            target[0:target.shape[0]//2] =0
            target[target.shape[0]//2:]=1
            
            #make masks there are 256/2 of them each has 10 patches out of 14*14 unmasked
            mask_zeros = torch.zeros(b_size, 14*14)
            mask_ones = torch.multinomial(torch.ones(b_size , 14*14 ), k_patch) ##hardocded k = 10 here
            index_x = np.repeat(np.arange(0, b_size ,1), k_patch)
            mask_ones = rearrange(mask_ones, 'b h  -> (b  h)')
            mask_zeros[index_x, mask_ones] = 1
            mask_zeros = rearrange(mask_zeros, 'b h  -> (b  h)')
            mask = torch.unsqueeze(mask_zeros,1)
            mask = torch.unsqueeze(mask,1)
            mask = torch.unsqueeze(mask,1)
            mask =  mask.to(device, non_blocking=True)

            image_ones = mask*(patches) +(1-mask)*rand_patches
            half_patches = patches.shape[0] //2
            images = torch.cat((rand_patches[0:half_patches], image_ones[half_patches:]  ),0)
            images =  rearrange(images, '(b h1 w1) c h w  -> b c (h1 h) (w1 w) ', h1=14, w1=14)
        



        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
