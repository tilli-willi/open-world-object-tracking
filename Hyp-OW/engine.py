# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# -----------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
 
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
 
import numpy as np
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.open_world_eval import OWEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from util.plot_utils import plot_prediction
import matplotlib.pyplot as plt
from copy import deepcopy
from datetime import datetime
from pytz import timezone
from PIL import Image
import sys
import pdb
import os.path as osp

from pathlib import Path
import json
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, nc_epoch: int, max_norm: float = 0, wandb: object = None,args=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
  
        
    format = "%Y-%m-%d %H:%M:%S "
    dt_utcnow = datetime.now(timezone('America/Los_Angeles'))
    header = ' \n {} Epoch: [{}]'.format(dt_utcnow.strftime(format),epoch)
    print_freq = 10
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    
    counter=0
    last_loss = 0
    for _ in metric_logger.log_every(range(len(data_loader)), args.logging_freq, header):
        counter+=1
        outputs = model(samples)
        # ForkedPdb().set_trace() 
        if args.model_type=='prob':
            loss_dict = criterion(outputs, targets) 
       
        elif args.model_type=='hypow':
            loss_dict = criterion(outputs, targets,counter,epoch)
       
                    
        # loss_dict = criterion(outputs, targets) 
        weight_dict = deepcopy(criterion.weight_dict)
        
       

        
        if epoch < nc_epoch: 
            for k,v in weight_dict.items():
                if 'NC' in k:
                    weight_dict[k] = 0
        
        #ForkedPdb().set_trace()
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        #ForkedPdb().set_trace()
        loss_dict_reduced = utils.reduce_dict(loss_dict)
     
        
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
 
        loss_value = losses_reduced_scaled.item()
 
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
 
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        try:
            optimizer.step()
        except:
            ForkedPdb().set_trace()
            
        if wandb is not None:
            wandb.log({"total_loss":loss_value})
            wandb.log(loss_dict_reduced_scaled)
            wandb.log(loss_dict_reduced_unscaled)
#
        
            
        
        
        
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        #
        
            
            
        
        samples, targets = prefetcher.next()
        
        if args.debug_epoch and counter==2:
            break
    # gather the stats from all processes
    
    metric_logger.synchronize_between_processes()
   
    print("Averaged stats:", metric_logger)
    
   
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def predict_custom(model, criterion, postprocessors, device, video_set_path, detections_path,
                   video_id=None, remove_background=False, pred_per_im=100):
    model.eval()
    criterion.eval()
    
    video_set_path = Path(video_set_path)
    video_ids = [video_id] if video_id is not None else [os.path.basename(str(f)) for f in video_set_path.iterdir() if f.is_dir()]
    
    from datasets.coco import make_coco_transforms
    from util.misc import nested_tensor_from_tensor_list
    import time
    
    transforms=make_coco_transforms('burst_val_test')
    videos_processed = 0
    videos_total = len(video_ids)
    inference_start_time = time.time()
    for video_id in video_ids:
        frame_paths = [str(f) for f in (video_set_path / video_id).iterdir() if f.is_file()]
        results = {}
        
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert('RGB')
            w, h = img.size
            img, _ = transforms[-1](img, None)
            
            img_as_nested_tensor = nested_tensor_from_tensor_list([img]).to(device)
            output = model(img_as_nested_tensor)
            orig_target_sizes = torch.stack([torch.as_tensor([int(h), int(w)]).to(device) for i in range(1)], dim=0)
            result = postprocessors['bbox'](output, orig_target_sizes, remove_background, pred_per_im)
            for i in range(1): 
                results.update({os.path.basename(frame_path): {key: value.cpu().numpy().tolist() for key, value in result[i].items()}})
        
        end_time = time.time()
        
        Path(detections_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(detections_path, video_id + ".json"), 'w') as f:
            json.dump(results, f, indent=4)
        videos_processed += 1
        print(f"Processed videos {videos_processed}/{videos_total}")
        print(f"Time passed: {(time.time() - inference_start_time):.4f} seconds, Avg time per video: {(time.time() - inference_start_time) / videos_processed:.4f} seconds")
    return


@torch.no_grad()
def predict_burst(model, criterion, postprocessors, device, dataset_name, 
                   detections_path, burst_annot_path, tao_frames_path,
                   args, burst_subset="val", video_id=None, remove_background=False, pred_per_im=100):
    model.eval()
    criterion.eval()
    
    burst_gt_path = os.path.join(burst_annot_path, burst_subset, "all_classes.json")

    with open(burst_gt_path, 'r') as file:
        burst_val_gt_raw = json.load(file)
    
    burst_val_gt = {f"{seq['dataset']}/{seq['seq_name']}": seq for seq in burst_val_gt_raw['sequences'] if seq['dataset'] == dataset_name}    
    video_ids = [video_id] if video_id is not None else [seq['seq_name'] for seq in burst_val_gt_raw['sequences'] if seq['dataset'] == dataset_name]
    
    from datasets.coco import make_coco_transforms
    from util.misc import nested_tensor_from_tensor_list
    import time
    
    transforms=make_coco_transforms('burst_val_test')
    videos_processed = 0
    videos_total = len(video_ids)
    inference_start_time = time.time()
    for video_id in video_ids:
        video_path = os.path.join(tao_frames_path, burst_subset, dataset_name, video_id)
        video_key = os.path.join(dataset_name, video_id)
        
        width, height = burst_val_gt[video_key]['width'], burst_val_gt[video_key]['height']
        results = {}
        num_frames = len(burst_val_gt[video_key]['all_image_paths'])
        num_processed = 0
        def batch_iterator_list(data, batch_size):
            for i in range(0, len(data), batch_size):
                yield data[i:i + batch_size]
        
        start_time = time.time()
        for img_path_batch in batch_iterator_list(burst_val_gt[video_key]['all_image_paths'], args.batch_size):            
            img_batch = []
            for img_path in img_path_batch:
                img = Image.open(os.path.join(video_path, img_path)).convert('RGB')
                img, _ = transforms[-1](img, None)
                img_batch.append(img)
            img_batch_as_nested_tensor = nested_tensor_from_tensor_list(img_batch).to(device)
            output = model(img_batch_as_nested_tensor)
            orig_target_sizes = torch.stack([torch.as_tensor([int(height), int(width)]).to(device) for i in range(len(img_batch))], dim=0)
            result = postprocessors['bbox'](output, orig_target_sizes, remove_background, pred_per_im)
            for i in range(len(img_batch)):            
                results.update({img_path_batch[i]: {key: value.cpu().numpy().tolist() for key, value in result[i].items()}})
            num_processed += len(img_batch)
            # print(f"Processed frame {num_processed}/{num_frames}")
            # print(f"Time passed: {(time.time() - start_time):.4f} seconds, Time per frame: {(time.time() - start_time) / num_processed:.4f} seconds")
        
        end_time = time.time()
        # print(f"Loop runtime: {(end_time - start_time):.4f} seconds")
        
        detections_for_dataset_path = os.path.join(detections_path, dataset_name)
        Path(detections_for_dataset_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(detections_for_dataset_path, video_id + ".json"), 'w') as f:
            json.dump(results, f, indent=4)
        videos_processed += 1
        print(f"Processed videos {videos_processed}/{videos_total}")
        print(f"Time passed: {(time.time() - inference_start_time):.4f} seconds, Avg time per video: {(time.time() - inference_start_time) / videos_processed:.4f} seconds")
    return

## ORIGINAL FUNCTION
@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args,remove_background=False,pred_per_im=100,temp=None):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = OWEvaluator(base_ds, iou_types, args=args)
 
    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
     
    relevant_matrix=None
    
    all_det = {} ## edit 1
    # # inference for a specific video /home/uig93971/src/data/TAO/frames/val/BDD/b2a5baf7-58519386/frame0001.jpg
    # burst_val_gt_path = '/home/uig93971/src/data/TAO/burst_annotations/val/all_classes.json'

    # with open(burst_val_gt_path, 'r') as file:
    #     burst_val_gt = json.load(file)
    
    # burst_val_gt = {f"{seq['dataset']}/{seq['seq_name']}": seq for seq in burst_val_gt['sequences']}
    
    # from datasets.coco import make_coco_transforms
    # from util.misc import nested_tensor_from_tensor_list
    # transforms=make_coco_transforms('burst_val_test')
    
    # dataset_name = "Charades"
    # video_id = "5TM3H"
    # video_path = os.path.join("/home/uig93971/src/data/TAO/frames/val", dataset_name, video_id)
    # video_key = os.path.join(dataset_name, video_id)
    
    # width, height = burst_val_gt[video_key]['width'], burst_val_gt[video_key]['height']
    # results = {}
    # num_frames = len(burst_val_gt[video_key]['all_image_paths'])
    # i = 1
    # for img_path in burst_val_gt[video_key]['all_image_paths']:
    #     img = Image.open(os.path.join(video_path, img_path)).convert('RGB')
    #     img, _ = transforms[-1](img, None)
    #     img = nested_tensor_from_tensor_list([img]).to(device)
    #     output = model(img)
    #     orig_target_sizes = torch.stack([torch.as_tensor([int(height), int(width)]).to(device)], dim=0)
    #     result = postprocessors['bbox'](output, orig_target_sizes,remove_background,pred_per_im)
    #     results.update({img_path: {key: value.cpu().numpy().tolist() for key, value in result[0].items()}})
    #     print(f"Processed frame {i}/{num_frames}")
    #     i += 1
    
    # detections_dir = os.path.join("/home/uig93971/src/Hyp-OW-burst-eval/detections", dataset_name)
    # Path(detections_dir).mkdir(parents=True, exist_ok=True)
    # with open(os.path.join(detections_dir, video_id + ".json"), 'w') as f:
    #     json.dump(results, f, indent=4)
    # return
    
    
    # img_path = "/home/uig93971/src/data/TAO/frames/val/BDD/b2a5baf7-58519386/frame0001.jpg"
    # img = Image.open(img_path).convert('RGB')
    # width, height = img.size
    # print(f"Image width: {width}, Image height: {height}")
    # img, _ = transforms[-1](img, None)
    # # print(img.shape)
    # # print(img)
    # img = nested_tensor_from_tensor_list([img]).to(device)
    # output = model(img)
    # orig_target_sizes = torch.stack([torch.as_tensor([int(width), int(height)]).to(device)], dim=0)
    # result = postprocessors['bbox'](output, orig_target_sizes,remove_background,pred_per_im)
    # # print("result")
    # # print(result)
    # return
    
       
    # it = 0
    for samples, targets in metric_logger.log_every(data_loader, 1, header):
        # it += 1
        # if it > 20:
        #     break
        print(samples.tensors.shape)
        print(samples.tensors[0])
        # return
        samples = samples.to(device)        
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
      
        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        if args.model_type=='hypow':
            results = postprocessors['bbox'](outputs, orig_target_sizes,remove_background,pred_per_im)
        else:
            results = postprocessors['bbox'](outputs, orig_target_sizes)
       
 
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        all_det = all_det | res ## edit 2
        if coco_evaluator is not None:
            coco_evaluator.update(res)
 
        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name
 
            panoptic_evaluator.update(res_pano)
        if args.debug_eval:
            break
 
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()
 
    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        res = coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['metrics']=res
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
        
    all_det_jsonified = {}
    for k, v in all_det.items():
        converted_dict = {key: value.cpu().numpy().tolist() for key, value in v.items()}
        all_det_jsonified[k] = converted_dict
    if args.save_eval_det_file: ## edit 3
        with open(args.save_eval_det_file, 'w') as f: ## edit 3
            json.dump(all_det_jsonified, f, indent=4) ## edit 3
    return stats, coco_evaluator
 
    
@torch.no_grad()
def get_exemplar_replay(model, exemplar_selection, device, data_loader):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = '[ExempReplay]'
    print_freq = 10
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    image_sorted_scores_reduced={}
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        image_sorted_scores = exemplar_selection(samples, outputs, targets)
        for i in utils.combine_dict(image_sorted_scores):
            image_sorted_scores_reduced.update(i[0])
            
        metric_logger.update(loss=len(image_sorted_scores_reduced.keys()))
        samples, targets = prefetcher.next()
        
    print(f'found a total of {len(image_sorted_scores_reduced.keys())} images')
    return image_sorted_scores_reduced