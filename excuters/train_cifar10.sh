SCRIPT=`realpath $0`
THISPATH=`dirname $SCRIPT`

DATASET_NAME=cifar10
DATASET_DIR=/home/cideep/Work/tensorflow/datasets/cifar-10/tfrecord

MODEL_NAME=inception_v4
CHECKPOINT_PATH=/home/cideep/Work/tensorflow/checkpoints/inception_v4.ckpt
CHECKPOINT_EXCLUDE='InceptionV4/Logits,InceptionV4/AuxLogits'

TRAIN_DIR=/home/cideep/Work/tensorflow/checkpoints/inception-v4-cifar10
rm -r ${TRAIN_DIR}
mkdir ${TRAIN_DIR}

echo 'start retraining inception-v4 on cifar10'
python ${THISPATH}/../train_image_classifier.py \
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

echo -e 'finished retraining inception-v4 on cifar10\n\n\n'

