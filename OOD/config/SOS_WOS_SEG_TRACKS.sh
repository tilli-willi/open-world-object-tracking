echo running track segmentation of SOS and WOS
set -x

PY_ARGS=${@:1}
python -u segment_ood.py \
    --segment_ood_tracklets \
    --video_set_path "/home/uig93971/src/data/street_obstacle_sequences/raw_data"\
    --track_res_path "/home/uig93971/src/data/street_obstacle_sequences/tracking_res/track_thresh0.1_dti"\
    ${PY_ARGS}

# PY_ARGS=${@:1}
# python -u segment_ood.py \
#     --video_set_path "/home/uig93971/src/data/wos/raw_data"\
#     --detections_path "/home/uig93971/src/data/wos/detections"\
#     --heatmap_path "/home/uig93971/src/data/wos/detections/heatmaps"\
#     ${PY_ARGS}
