SCRIPT=`realpath $0`
THISPATH=`dirname $SCRIPT`

DATASET_NAME=cifar10
DATASET_DIR=/home/cideep/Work/tensorflow/datasets/cifar-10/tfrecord

MODEL_NAME=inception_resnet_v2
CHECKPOINT_PATH=/home/cideep/Work/tensorflow/checkpoints/my-fine-tuned/inc-resnet-v2-cifar10

EVAL_DIR=${CHECKPOINT_PATH}/result
rm -r ${EVAL_DIR}
mkdir ${EVAL_DIR}

echo 'start retraining inception_resnet_v2 on cifar10'
python ${THISPATH}/../eval_image_classifier.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=test \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${CHECKPOINT_PATH}

echo -e 'finished retraining inception_resnet_v2 on cifar10\n\n\n'

