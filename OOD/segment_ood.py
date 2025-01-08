import argparse
from copy import deepcopy
import json
import os
import os.path as osp
from pathlib import Path
import time

import numpy as np
import torch
import pandas as pd
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from pycocotools import mask as cocomask


def make_parser():
    parser = argparse.ArgumentParser("Segment OOD")
    parser.add_argument('--video_id', type=str, help='choose specific video id to run inference on, if None, run on all videos')
    parser.add_argument('--video_set_path', default=None, type=str, help='path to the video set')
    parser.add_argument('--heatmap_path', default=None, type=str, help='path to save segmentation results')
    parser.add_argument('--detections_path', default=None, type=str, help='path to detections as boxes')
    parser.add_argument('--sam_model_size', type=str, choices=['tiny', 'small', 'base_plus', 'large'], help='choose SAM model size')
    parser.add_argument('--segmentation_setup_name', default=None, type=str, help='name of the segmentation setup')
    parser.add_argument('--segment_custom_video_set', default=False, action='store_true', help='generate segmentations and heatmap for custom dataset')
    
    parser.add_argument('--segment_ood_tracklets', default=False, action='store_true', help='generate segmentations, heatmap and track masks for box tracklets')
    parser.add_argument('--track_res_path', default=None, type=str, help='path to box tracklets')
    
    return parser


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

def preprocess_detections(detections, iou_threshold = 0.7, 
                          scale_for_known = 1, scale_for_unknown = 1):
    boxes_scores = [detections["boxes"][i] + [detections["scores"][i]] for i in range(len(detections["boxes"]))]
    boxes_scores_as_tensor = torch.tensor(boxes_scores, dtype=torch.float32, device='cpu')
    cls_as_tensor = torch.tensor(detections["labels"], dtype=torch.int64, device='cpu')
    unknown_class = 80
    keep_indices = custom_nms(boxes_scores_as_tensor[:, :4], boxes_scores_as_tensor[:, 4], cls_as_tensor, unknown_class, iou_threshold)
    final_det = []
    for i in range(len(detections["boxes"])):
        if i in keep_indices:
            new_score = detections["scores"][i]
            if detections["labels"][i] == unknown_class:
                new_score = min(1, new_score * scale_for_unknown)
            else:
                new_score = min(1, new_score * scale_for_known)
            final_det.append(detections["boxes"][i] + [new_score] + [detections["labels"][i]])
    final_det.sort(key=lambda x: x[-2], reverse=True)
    return np.array(final_det, dtype=np.float64)

from torchvision.ops.boxes import box_iou

def custom_nms(boxes, scores, labels, target_class, iou_threshold, high_iou_threshold=0.85):
    """
    Perform custom NMS where the target class is suppressed by all overlapping boxes.
    
    Args:
        boxes (Tensor): Tensor of shape (N, 4) with bounding box coordinates.
        scores (Tensor): Tensor of shape (N,) with confidence scores.
        labels (Tensor): Tensor of shape (N,) with class labels.
        target_class (int): The class to be suppressed by all overlaps.
        iou_threshold (float): IoU threshold for suppression.
        high_iou_threshold (float): IoU threshold for suppression of almost identical boxes regardless of class.
    Returns:
        keep_indices (Tensor): Indices of boxes to keep after suppression.
    """
    keep = torch.ones(len(boxes), dtype=torch.bool)  # Initialize keep mask

    # Compute pairwise IoU for all boxes
    iou_matrix = box_iou(boxes, boxes)

    for i in range(len(boxes)):
        if not keep[i]:
            continue  # Skip suppressed boxes
        
        # Check if the current box is the target class
        if labels[i] == target_class:
            suppress_mask = (iou_matrix[i] > iou_threshold) & (labels == target_class) & (scores[i] > scores) 
            keep[suppress_mask] = False
        else:
            suppress_mask = ((iou_matrix[i] > iou_threshold) & (labels == labels[i]) & (scores[i] > scores)) | \
                            ((iou_matrix[i] > iou_threshold) & (labels == target_class) & (scores[i] > 0.5)) | \
                            ((iou_matrix[i] > high_iou_threshold) & (scores[i] > scores))
            keep[suppress_mask] = False

    return torch.where(keep)[0]  # Indices of boxes to keep

from scipy.ndimage import distance_transform_edt

def find_centroid(mask):
    """
    Find the centroid (center of mass) of the segmentation mask.
    Ensures the chosen pixel is within the mask.
    """
    y_indices, x_indices = np.where(mask)
    centroid_x = int(np.mean(x_indices))
    centroid_y = int(np.mean(y_indices))
    
    # Check if the centroid is within the mask; if not, find the nearest valid pixel
    if not mask[centroid_y, centroid_x]:
        return find_nearest_valid_pixel(mask, (centroid_y, centroid_x))
    return centroid_x, centroid_y

def find_distance_transform_center(mask):
    """
    Find the center using a distance transform.
    Ensures the chosen pixel is within the mask.
    """
    distance = distance_transform_edt(mask)
    max_dist_idx = np.unravel_index(np.argmax(distance), mask.shape)
    return max_dist_idx  # This will always be inside the mask by definition

def find_nearest_valid_pixel(mask, point):
    """
    Find the nearest valid pixel in the mask to a given point.
    """
    distance = distance_transform_edt(mask == 0)  # Compute distance to non-mask regions
    nearest_idx = np.unravel_index(np.argmin(distance), mask.shape)
    return nearest_idx

def filter_masks(predictor, masks, scores, track_ids, area_inc_threshold=2000):
    filtered_masks = []
    filtered_scores = []
    filtered_track_ids = []
    for mask, score, track_id in zip(masks, scores, track_ids):
        ref_point = np.array(find_centroid(mask))
        refined_mask, _, _ = predictor.predict(
            point_coords=[ref_point],
            point_labels=np.array([1]),
            multimask_output=False,
        )
        refined_mask = refined_mask.squeeze()
        refined_area = np.sum(refined_mask)
        orig_area = np.sum(mask)
        # print(mask)
        intersection_area = np.sum(np.array(refined_mask, dtype=bool) & np.array(mask, dtype=bool))
        if not (intersection_area / orig_area > 0.8 and refined_area - orig_area > area_inc_threshold): 
            filtered_masks.append(mask)
            filtered_scores.append(float(score))
            filtered_track_ids.append(track_id)
            
    return filtered_masks, filtered_scores, filtered_track_ids


def get_bounding_box(mask):
    """
    Calculate the bounding box of an object from a boolean segmentation mask.
    
    Parameters:
        mask (numpy.ndarray): Boolean numpy array of shape (h, w) representing the segmentation mask.

    Returns:
        tuple: Bounding box coordinates (x_min, y_min, x_max, y_max).
    """
    # Get the indices of True values in the mask
    rows, cols = np.where(mask)
    
    # If there are no True values, return None
    if len(rows) == 0 or len(cols) == 0:
        return None
    
    # Calculate the bounding box
    x_min, x_max = cols.min(), cols.max()
    y_min, y_max = rows.min(), rows.max()
    
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def mask_to_rle_ann(mask: np.ndarray):
    assert mask.ndim == 2, f"Mask must be a 2-D array, but got array of shape {mask.shape}"
    rle = cocomask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def segment_custom_video_set(video_set_path, detections_path, heatmap_path, video_id=None):
    video_set_path = Path(video_set_path)
    video_ids = [video_id] if video_id is not None else [os.path.basename(str(f)) for f in video_set_path.iterdir() if f.is_dir()]
    
    ##############################
    device = torch.device("cuda")
    
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
            
    # Model initialization
    sam2_checkpoint = f"/home/uig93971/src/open-world-object-tracking/sam2/checkpoints/sam2.1_hiera_large.pt" # TODO add to arguments!
    model_cfg = f"configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    ##############################
    
    videos_processed = 0
    videos_total = len(video_ids)
    inference_start_time = time.time()
    
    for video_id in video_ids:
        frame_paths = [str(f) for f in (video_set_path / video_id).iterdir() if f.is_file()]
        frame_paths.sort()
        det_for_video = json.load(open(os.path.join(detections_path, video_id + ".json")))
        Path(osp.join(heatmap_path, video_id)).mkdir(parents=True, exist_ok=True)
        Path(osp.join(detections_path, "boxes_masks", video_id)).mkdir(parents=True, exist_ok=True)
        for frame_path in frame_paths:
            image = Image.open(frame_path)
            w, h = image.size
            image = np.array(image.convert("RGB"))
            predictor.set_image(image)

            objectness_threshold = 0.1 # TODO add to arguments
            seg_threshold = 0.7 # TODO add to arguments
                    
            det_preproc = preprocess_detections(det_for_video[os.path.basename(frame_path)])
            input_boxes = np.array([box for box in det_preproc 
                                    if (box[-2] > objectness_threshold) 
                                    and (abs(box[0] - box[2]) < 0.4 * w) 
                                    and (abs(box[1] - box[3]) < 0.7 * h)])[:, 0:-2] # TODO add to arguments


            masks, scores, _ = predictor.predict(
                box=input_boxes,
                multimask_output=False,
            )

            masks = masks.squeeze()
            scores = scores.squeeze()
            filtered_masks = np.array(masks)[scores.squeeze() > seg_threshold]
            filtered_scores = scores.squeeze()[scores.squeeze() > seg_threshold]
            filtered_masks, filtered_scores = filter_masks(predictor, filtered_masks, filtered_scores)
            frame_entry = {'boxes': [], 'scores': [], 'masks': []}
            if len(filtered_masks) > 0:
                filtered_masks = unoverlap_masks(filtered_masks, np.array(filtered_scores))
                boxes = [get_bounding_box(mask) for mask in filtered_masks]
                enc_masks = [mask_to_rle_ann(mask)['counts'] for mask in filtered_masks]
                heatmap = np.sum(np.array(filtered_masks) * np.array(filtered_scores)[:, np.newaxis, np.newaxis], axis=0)
            else:
                heatmap = np.zeros((h, w), dtype=np.float32)
            frame_entry['boxes'] = boxes
            frame_entry['scores'] = filtered_scores
            frame_entry['masks'] = enc_masks
            with open(os.path.join(detections_path, "boxes_masks", video_id, osp.splitext(os.path.basename(frame_path))[0] + ".json"), 'w') as f:
                json.dump(frame_entry, f, indent=4)
            np.save(os.path.join(heatmap_path, video_id, osp.splitext(os.path.basename(frame_path))[0] + ".npy"), heatmap)
        
        videos_processed += 1
        print(f"Processed videos {videos_processed}/{videos_total}")
        print(f"Time passed: {(time.time() - inference_start_time):.4f} seconds, Avg time per video: {(time.time() - inference_start_time) / videos_processed:.4f} seconds")
    return


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


def segment_ood_tracklets(video_set_path, track_res_path, video_id=None):
    video_set_path = Path(video_set_path)
    video_ids = [video_id] if video_id is not None else [os.path.basename(str(f)) for f in video_set_path.iterdir() if f.is_dir()]
    
    ##############################
    device = torch.device("cuda")
    
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
            
    # Model initialization
    sam2_checkpoint = f"/home/uig93971/src/open-world-object-tracking/sam2/checkpoints/sam2.1_hiera_large.pt" # TODO add to arguments!
    model_cfg = f"configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    ##############################
    
    videos_processed = 0
    videos_total = len(video_ids)
    inference_start_time = time.time()
    
    for video_id in video_ids:
        frame_paths = [str(f) for f in (video_set_path / video_id).iterdir() if f.is_file()]
        frame_paths.sort()
        tracklet_dict = get_tracklets_for_video(osp.join(track_res_path, video_id))
        Path(osp.join(track_res_path, "heatmaps", video_id)).mkdir(parents=True, exist_ok=True)
        Path(osp.join(track_res_path, "boxes_masks", video_id)).mkdir(parents=True, exist_ok=True)
        Path(osp.join(track_res_path, "ood_prediction_tracked", video_id)).mkdir(parents=True, exist_ok=True)
        for frame_id, frame_path in enumerate(frame_paths, 1):
            image = Image.open(frame_path)
            w, h = image.size
            image = np.array(image.convert("RGB"))
            predictor.set_image(image)

            # objectness_threshold = 0.1 # TODO add to arguments
            seg_threshold = 0.7 # TODO add to arguments
            
            cur_frame = tracklet_dict.get(frame_id, None)
            cur_frame = [box for box in cur_frame 
                         if (box[3] < 0.4 * w) 
                         and (box[4] < 0.7 * h)] # TODO add to arguments
            track_ids = np.array(cur_frame)[:, 0]
            input_boxes = np.array(cur_frame)[:, 1:-1]
            input_boxes[:, 2:] = input_boxes[:, 2:] + input_boxes[:, :2]
            
            masks, scores, _ = predictor.predict(
                box=input_boxes,
                multimask_output=False,
            )

            masks = masks.squeeze()
            scores = scores.squeeze()
            filtered_masks = np.array(masks)[scores > seg_threshold]
            filtered_scores = scores[scores > seg_threshold]
            filtered_track_ids = track_ids[scores > seg_threshold]
            filtered_masks, filtered_scores, filtered_track_ids = filter_masks(predictor, filtered_masks, filtered_scores, filtered_track_ids)
            frame_entry = {'seg_boxes': [], 'scores': [], 'masks': [], 'track_ids': []}
            if len(filtered_masks) > 0:
                filtered_masks = unoverlap_masks(filtered_masks, np.array(filtered_scores))
                # Remove empty masks
                filtered_masks_non_empty = []
                filtered_scores_non_empty = []
                filtered_track_ids_non_empty = []
                for mask, score, track_id in zip(filtered_masks, filtered_scores, filtered_track_ids):
                    if np.sum(mask) > 0:
                        filtered_masks_non_empty.append(mask)
                        filtered_scores_non_empty.append(score)
                        filtered_track_ids_non_empty.append(track_id)
                        
                seg_boxes = [get_bounding_box(mask) for mask in filtered_masks_non_empty]
                enc_masks = [mask_to_rle_ann(mask)['counts'] for mask in filtered_masks_non_empty]
                heatmap = np.sum(np.array(filtered_masks_non_empty) * np.array(filtered_scores_non_empty)[:, np.newaxis, np.newaxis], axis=0)
                track_mask = np.sum(np.array(filtered_masks_non_empty) * np.array(filtered_track_ids_non_empty, dtype=int)[:, np.newaxis, np.newaxis], axis=0)
            else:
                heatmap = np.zeros((h, w), dtype=np.float32)
            frame_entry['seg_boxes'] = seg_boxes
            frame_entry['scores'] = filtered_scores
            frame_entry['masks'] = enc_masks
            frame_entry['track_ids'] = filtered_track_ids_non_empty
            with open(os.path.join(track_res_path, "boxes_masks", video_id, osp.splitext(os.path.basename(frame_path))[0] + ".json"), 'w') as f:
                json.dump(frame_entry, f, indent=4)
            np.save(os.path.join(track_res_path, "heatmaps", video_id, osp.splitext(os.path.basename(frame_path))[0] + ".npy"), heatmap)
            np.save(os.path.join(track_res_path, "ood_prediction_tracked", video_id, osp.splitext(os.path.basename(frame_path))[0] + ".npy"), track_mask)
        
        videos_processed += 1
        print(f"Processed videos {videos_processed}/{videos_total}")
        print(f"Time passed: {(time.time() - inference_start_time):.4f} seconds, Avg time per video: {(time.time() - inference_start_time) / videos_processed:.4f} seconds")
    return





if __name__ == "__main__":
    args = make_parser().parse_args()

    if args.segment_custom_video_set:
        print("Segmenting custom video set")
        segment_custom_video_set(args.video_set_path, args.detections_path, args.heatmap_path)

    if args.segment_ood_tracklets:
        print("Segmenting OOD tracklets")
        segment_ood_tracklets(args.video_set_path, args.track_res_path)
    
    