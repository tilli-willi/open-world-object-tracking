echo running segmentation of SOS and WOS
set -x

PY_ARGS=${@:1}
python -u segment_ood.py \
    --video_set_path "/home/uig93971/src/data/street_obstacle_sequences/raw_data"\
    --detections_path "/home/uig93971/src/data/street_obstacle_sequences/detections"\
    --heatmap_path "/home/uig93971/src/data/street_obstacle_sequences/detections/heatmaps"\
    ${PY_ARGS}

PY_ARGS=${@:1}
python -u segment_ood.py \
    --video_set_path "/home/uig93971/src/data/wos/raw_data"\
    --detections_path "/home/uig93971/src/data/wos/detections"\
    --heatmap_path "/home/uig93971/src/data/wos/detections/heatmaps"\
    ${PY_ARGS}
