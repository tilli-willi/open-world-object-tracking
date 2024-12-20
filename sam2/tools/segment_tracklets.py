import argparse
from copy import deepcopy
import json
import os
import os.path as osp
import time

import numpy as np
import torch
import pandas as pd
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from pycocotools import mask as cocomask


def make_parser():
    parser = argparse.ArgumentParser("SAM2: Segment tracklets")
    parser.add_argument('--burst_subdataset', type=str, help='choose BURST subdataset to run inference on (e.g. ArgoVerse, BDD, Charades, LaSOT, YFCC100M)')
    parser.add_argument('--burst_subset', type=str, help='choose BURST subset to run inference on (train, val, test)')
    parser.add_argument('--video_id', type=str, help='choose specific video id to run inference on, if None, run on all videos')
    parser.add_argument('--detections_path', default=None, type=str, help='path to save detections')
    parser.add_argument('--burst_annot_path', default=None, type=str, help='path to BURST annotations')
    parser.add_argument('--track_results_path', default=None, type=str, help='path to tracking results')
    parser.add_argument('--tracking_exp_name', default=None, type=str, help='name of the tracking experiment')
    parser.add_argument('--seg_results_path', default=None, type=str, help='path to segmented tracking results')
    parser.add_argument('--sam_model_size', type=str, choices=['tiny', 'small', 'base_plus', 'large'], help='choose SAM model size')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size for inference')
    parser.add_argument('--segmentation_setup_name', default=None, type=str, help='name of the segmentation setup')
    parser.add_argument('--tao_frames_path', default=None, type=str, help='path to TAO frames')
    return parser

from itertools import islice

def batched_zip(list1, list2, batch_size):
    """
    Iterate over two lists in a batched manner.
    
    Args:
        list1 (list): First list.
        list2 (list): Second list.
        batch_size (int): Size of each batch.

    Yields:
        tuple: Batches of paired elements from the two lists.
    """
    it1, it2 = iter(list1), iter(list2)
    while True:
        batch1 = list(islice(it1, batch_size))
        batch2 = list(islice(it2, batch_size))
        if not batch1 or not batch2:
            break
        yield batch1, batch2

def get_tracklets_for_video(path):
    tracklets = pd.read_csv(path + ".txt", header=None)
    tracklets.columns = ["frame_id", 
                     "object_id", 
                     "b_t", 
                     "b_l", 
                     "b_w", 
                     "b_h", 
                     "confidence", 
                     "Col8", 
                     "Col9", 
                     "Col10"]

    tracklet_dict = {}
    for _, row in tracklets.iterrows():
        frame_id = int(row['frame_id'])
        obj_data = row[['object_id', "b_t", "b_l", "b_w", "b_h", 'confidence']].tolist()
        obj_data[0] = int(obj_data[0])
        if frame_id not in tracklet_dict:
            tracklet_dict[frame_id] = []
        tracklet_dict[frame_id].append(obj_data)
    return tracklet_dict

def unoverlap_masks(masks, scores):
    """
    Post-process segmentation masks to make them non-overlapping,
    giving priority to higher scoring masks.
    
    Args:
        masks (list of np.ndarray): List of binary segmentation masks (each of shape [H, W]).
        scores (np.ndarray): Array of scores corresponding to each mask.
        
    Returns:
        list of np.ndarray: List of updated, non-overlapping binary masks.
    """
    # Sort masks and scores by descending scores
    sorted_indices = np.argsort(scores.squeeze())[::-1]
    sorted_masks = [masks[i] for i in sorted_indices]
    
    # Create an empty canvas to track which pixels are already assigned
    occupied = np.zeros_like(sorted_masks[0], dtype=bool)
    
    # Create the list to store non-overlapping masks
    non_overlapping_masks = []
    
    for mask in sorted_masks:
        # Convert the mask to boolean to ensure compatibility
        mask = mask.astype(bool)
        
        # Remove overlaps by keeping only unoccupied areas
        non_overlapping_mask = mask & ~occupied
        non_overlapping_masks.append(non_overlapping_mask)
        
        # Update the occupied canvas
        occupied |= non_overlapping_mask  # Now works correctly since types match
    
    # Reorder masks to their original order
    reordered_masks = [non_overlapping_masks[i] for i in np.argsort(sorted_indices)]
    
    return reordered_masks


def mask_to_rle_ann(mask: np.ndarray):
    assert mask.ndim == 2, f"Mask must be a 2-D array, but got array of shape {mask.shape}"
    rle = cocomask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def prepare_pred_entry(seq):
    prepared_seq = deepcopy(seq)
    prepared_seq["neg_category_ids"] = None
    prepared_seq["not_exhaustive_category_ids"] = None
    prepared_seq["segmentations"] = []
    prepared_seq["track_category_ids"] = {}
    return prepared_seq
    

def generate_segmented_tracklets(tracking_res_path, tracking_exp_name, dataset_name, 
                   burst_annot_path, tao_frames_path, seg_setup_name, args,
                   batch_size, sam_model_size, burst_subset="val", video_id=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")
    
    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
    # Model initialization
    sam2_checkpoint = f"/home/uig93971/src/sam2/checkpoints/sam2.1_hiera_{sam_model_size}.pt" # TODO add to arguments!
    model_cfg = f"configs/sam2.1/sam2.1_hiera_{sam_model_size[0]}.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    ##############################
            
    burst_gt_path = os.path.join(burst_annot_path, burst_subset, "all_classes.json")

    with open(burst_gt_path, 'r') as file:
        burst_val_gt_raw = json.load(file)
    
    burst_val_gt = {f"{seq['dataset']}/{seq['seq_name']}": seq for seq in burst_val_gt_raw['sequences'] if seq['dataset'] == dataset_name}    
    video_ids = [video_id] if video_id is not None else [seq['seq_name'] for seq in burst_val_gt_raw['sequences'] if seq['dataset'] == dataset_name]
    
    final_res_dict = deepcopy(burst_val_gt_raw)
    final_res_dict['sequences'] = []
    final_res_dict['categories'] = None
    final_res_dict['class_split'] = None
    
    videos_processed = 0
    videos_total = len(video_ids)
    inference_start_time = time.time()
    print(f"Starting inference for {dataset_name}...")
    for video_id in video_ids:
        video_path = os.path.join(tao_frames_path, burst_subset, dataset_name, video_id)
        video_key = os.path.join(dataset_name, video_id)
        tracklet_dict = get_tracklets_for_video(osp.join(tracking_res_path, tracking_exp_name, burst_subset, dataset_name, video_id))
        pred_entry = prepare_pred_entry(burst_val_gt[video_key])
        video_fps = burst_val_gt[video_key]['fps']
        frame_id = 0
        start_time = time.time()
        for frame_id, img_path in enumerate(burst_val_gt[video_key]['all_image_paths'], 1):
            if frame_id % video_fps == 1:
                image = Image.open(os.path.join(video_path, img_path))
                image = np.array(image.convert("RGB"))
                predictor.set_image(image)
                
                cur_frame = tracklet_dict.get(frame_id, None)
                frame_seg = {}
                if cur_frame is None:
                    pred_entry["segmentations"].append(frame_seg)
                    continue
                input_boxes = np.array(cur_frame)[:, 1:-1]
                input_boxes[:, 2:] = input_boxes[:, 2:] + input_boxes[:, :2]
                masks, scores, _ = predictor.predict(
                    box=input_boxes,
                    multimask_output=False,
                )
                            
                unoverlapped_masks = unoverlap_masks(masks, scores)
                for mask, obj_id in zip(unoverlapped_masks, np.array(cur_frame)[:, 0]):
                    pred_entry["track_category_ids"][str(int(obj_id))] = 0
                    frame_seg[str(int(obj_id))] = {"rle": mask_to_rle_ann(mask.squeeze())['counts'], "is_gt": False}
                pred_entry["segmentations"].append(frame_seg)  
                
                # print(f"Processed frame {frame_id}/{len(burst_val_gt[video_key]['all_image_paths'])}")
                # print(f"Time passed: {(time.time() - start_time):.4f} seconds, Time per frame: {(time.time() - start_time) / frame_id:.4f} seconds")
        
        assert len(pred_entry['annotated_image_paths']) == len(pred_entry['segmentations'])        
        final_res_dict['sequences'].append(pred_entry)
            
        videos_processed += 1
        print(f"Processed videos {videos_processed}/{videos_total}")
        print(f"Time passed: {(time.time() - inference_start_time):.4f} seconds, Avg time per video: {(time.time() - inference_start_time) / videos_processed:.4f} seconds")
    
    os.makedirs(osp.join(tracking_res_path, tracking_exp_name, burst_subset, "segmentations", seg_setup_name), exist_ok=True)
    res_file = osp.join(tracking_res_path, tracking_exp_name, burst_subset, "segmentations", seg_setup_name, f"{dataset_name}.json")        
    with open(res_file, 'w') as f:
        json.dump(final_res_dict, f, indent=4)
    
    
        
if __name__ == "__main__":
    args = make_parser().parse_args()

    generate_segmented_tracklets(args.track_results_path, args.tracking_exp_name, args.burst_subdataset, 
                   args.burst_annot_path, args.tao_frames_path, args.segmentation_setup_name, args,
                   args.batch_size, args.sam_model_size, video_id=args.video_id)