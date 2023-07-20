#!/bin/bash

#cd ../..

# custom config
DATA="./data"
TRAINER=VPT2

DATASET=$1
SEED=$2

CFG=vit_b16_ctx16_ep100_batch64_highLR

DIR=output/${DATASET}/${TRAINER}/${CFG}/seed${SEED}/scale1.5
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
    echo "Run this job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=4 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --crop-scale 1.5 \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAIN.CHECKPOINT_FREQ 2
fi