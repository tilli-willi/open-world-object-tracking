echo running tracking on SOS
set -x

PY_ARGS=${@:1}
python -u tools/track_custom_detections.py \
    --track_custom \
    --track_thresh 0.1 \
    --match_thresh 0.6 \
    --video_set_path "/home/uig93971/src/data/street_obstacle_sequences/raw_data"\
    --detections_path "/home/uig93971/src/data/street_obstacle_sequences/detections"\
    --track_results_path "/home/uig93971/src/data/street_obstacle_sequences/tracking_res"\
    ${PY_ARGS}


echo running tracking on WOS
python -u tools/track_custom_detections.py \
    --track_custom \
    --track_thresh 0.1 \
    --match_thresh 0.6 \
    --video_set_path "/home/uig93971/src/data/wos/raw_data"\
    --detections_path "/home/uig93971/src/data/wos/detections"\
    --track_results_path "/home/uig93971/src/data/wos/tracking_res"\
    ${PY_ARGS}
