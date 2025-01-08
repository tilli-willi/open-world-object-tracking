echo running segmentation of ObstacleTrack
set -x

PY_ARGS=${@:1}
python -u segment_ood.py \
    --video_set_path "/home/uig93971/src/data/dataset_ObstacleTrack"\
    --detections_path "/home/uig93971/src/data/dataset_ObstacleTrack/detections"\
    --heatmap_path "/home/uig93971/src/data/dataset_ObstacleTrack/detections/heatmaps"\
    ${PY_ARGS}

