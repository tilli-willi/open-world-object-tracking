import argparse
import json
import os
import os.path as osp
import time

import numpy as np
import cv2
import torch

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("-expn", "--experiment_name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument('--burst_subdataset', type=str, help='choose BURST subdataset to run inference on (e.g. ArgoVerse, BDD, Charades, LaSOT, YFCC100M)')
    parser.add_argument('--burst_subset', type=str, help='choose BURST subset to run inference on (train, val, test)')
    parser.add_argument('--video_id', type=str, help='choose specific video id to run inference on, if None, run on all videos')
    parser.add_argument('--detections_path', default=None, type=str, help='path to save detections')
    parser.add_argument('--burst_annot_path', default=None, type=str, help='path to BURST annotations')
    parser.add_argument('--track_results_path', default=None, type=str, help='path to tracking results')
    return parser


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def init_img_info(img_path):
    img_info = {"id": 0}
    img_info["file_name"] = img_path
    img = cv2.imread(img_path)

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img
    return img_info

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

def preprocess_detections(detections, iou_threshold = 0.5, 
                          scale_for_known = 1.5, scale_for_unknown = 0.7):
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
    

def track_custom(tracking_res_path, dataset_name, 
                 detections_path, burst_annot_path,
                 args, burst_subset="val", video_id=None):
    burst_gt_path = os.path.join(burst_annot_path, burst_subset, "all_classes.json")

    with open(burst_gt_path, 'r') as file:
        burst_val_gt_raw = json.load(file)
    
    burst_val_gt = {f"{seq['dataset']}/{seq['seq_name']}": seq for seq in burst_val_gt_raw['sequences'] if seq['dataset'] == dataset_name}    
    video_ids = [video_id] if video_id is not None else [seq['seq_name'] for seq in burst_val_gt_raw['sequences'] if seq['dataset'] == dataset_name]
    
    videos_processed = 0
    videos_total = len(video_ids)
    inference_start_time = time.time()
    for video_id in video_ids:
        video_key = os.path.join(dataset_name, video_id)
        det_path = os.path.join(detections_path, dataset_name, video_id + ".json")
        with open(det_path, 'r') as file:
            detections = json.load(file)    
        
        tracker = BYTETracker(args, frame_rate=args.fps)
        timer = Timer()
        results = []
        
        h = burst_val_gt[video_key]['height']
        w = burst_val_gt[video_key]['width']

        for frame_id, img_path in enumerate(burst_val_gt[video_key]['all_image_paths'], 1):
            outputs = [preprocess_detections(detections[img_path])]
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0][:, :5], [h, w], (h, w))
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    if tlwh[2] * tlwh[3] > args.min_box_area: # ignoring tiny boxes
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        # save results
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()

            # if frame_id % 20 == 0:
            #     logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        os.makedirs(osp.join(tracking_res_path, args.experiment_name, burst_subset, dataset_name), exist_ok=True)
        res_file = osp.join(tracking_res_path, args.experiment_name, burst_subset, dataset_name, f"{video_id}.txt")        
        with open(res_file, 'w') as f:
            f.writelines(results)
        
        videos_processed += 1
        print(f"Processed videos {videos_processed}/{videos_total}")
        print(f"Time passed: {(time.time() - inference_start_time):.4f} seconds, Avg time per video: {(time.time() - inference_start_time) / videos_processed:.4f} seconds")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    track_custom(args.track_results_path, args.burst_subdataset, 
                 args.detections_path, args.burst_annot_path, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)

    main(exp, args)
