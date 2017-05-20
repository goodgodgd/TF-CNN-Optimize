# changing lines
DATASET_NAME=cifar10
MODEL_NAME=inception_v4
SCOPE_VAR=InceptionV4

# below lines are fixed
PYTHON_PATH=/home/cideep/Work/tensorflow/tfenv/bin
SCRIPT=`realpath $0`
THISPATH=`dirname $SCRIPT`

DATASET_DIR=/home/cideep/Work/tensorflow/datasets/${DATASET_NAME}/tfrecord
CHECKPOINT_PATH=/home/cideep/Work/tensorflow/checkpoints/${MODEL_NAME}.ckpt
CHECKPOINT_EXCLUDE="${SCOPE_VAR}/Logits,${SCOPE_VAR}/AuxLogits"
TRAIN_DIR=/home/cideep/Work/tensorflow/checkpoints/${MODEL_NAME}_${DATASET_NAME}
rm -r ${TRAIN_DIR}
mkdir ${TRAIN_DIR}

echo -e "start retraining ${MODEL_NAME} on ${DATASET_NAME}"
${PYTHON_PATH}/python ${THISPATH}/../train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --optimizer=rmsprop \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --batch_size=32 \
    --learning_rate=0.01 \
    --learning_rate_decay_factor=0.8 \
    --max_number_of_steps=50000 \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=${CHECKPOINT_EXCLUDE}

echo -e "finished retraining ${MODEL_NAME} on ${DATASET_NAME}"

