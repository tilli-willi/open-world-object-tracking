#!/usr/bin/env bash

echo running eval of WOS

set -x

EXP_DIR=exps
PY_ARGS=${@:1}
WANDB_NAME=PROB_V1
 
PY_ARGS=${@:1}
python -u main_open_world.py \
    --predict_custom \
    --video_set_path "/home/uig93971/src/data/wos/raw_data" \
    --detections_path "/home/uig93971/src/data/wos/detections" \
    --dataset HIERARCHICAL --PREV_INTRODUCED_CLS 80 --CUR_INTRODUCED_CLS 0 \
    --train_set "" --test_set 'burst_val_test' --epochs 191 --lr_drop 35\
    --model_type 'hypow' --obj_loss_coef 8e-4 --obj_temp 1.3\
    --pretrain "${EXP_DIR}/hypow_t4_ft_hierarchical_split.pth" --eval --wandb_project ""\
      --wandb_name "${WANDB_NAME}_t1" --wandb_project '' --lr_drop 40  --num_queries 100 --logging_freq 40 --use_focal_cls  \
    --save_buffer  --relabel  --eval \
    --epochs 50  --clip_r 1.0 --use_hyperbolic --unknown_weight 0.1  \
    --hyperbolic_c 0.1 --hyperbolic_temp 0.2 --samples_per_category 1 --hyperbolic_coeff 0.05 --checkpoint_period 10 --start_relabelling 0 --emb_per_class 10 \
    --all_background --empty_weight 0.1 --use_max_uperbound  --family_regularizer --family_hyperbolic_coeff 0.02  \
    ${PY_ARGS}
