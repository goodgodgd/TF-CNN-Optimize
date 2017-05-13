SCRIPT=`realpath $0`
THISPATH=`dirname $SCRIPT`

DATASET_NAME=cifar100
DATASET_DIR=/home/cideep/Work/tensorflow/datasets/cifar-100/tfrecord

MODEL_NAME=inception_resnet_v2
CHECKPOINT_PATH=/home/cideep/Work/tensorflow/checkpoints/inception_resnet_v2_2016_08_30.ckpt
CHECKPOINT_EXCLUDE='InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits'

TRAIN_DIR=/home/cideep/Work/tensorflow/checkpoints/inc-resnet-v2-cifar100
rm -r ${TRAIN_DIR}
mkdir ${TRAIN_DIR}

echo 'start retraining inception-Resnet-v2 on cifar100'
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

echo -e 'finished retraining inception-Resnet-v2 on cifar100\n\n\n'

