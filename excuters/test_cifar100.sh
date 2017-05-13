SCRIPT=`realpath $0`
THISPATH=`dirname $SCRIPT`

DATASET_NAME=cifar100
DATASET_DIR=/home/cideep/Work/tensorflow/datasets/cifar-100/tfrecord

MODEL_NAME=inception_v4
CHECKPOINT_PATH=/home/cideep/Work/tensorflow/checkpoints/my-fine-tuned/inception-v4-cifar100

EVAL_DIR=${CHECKPOINT_PATH}/result
rm -r ${EVAL_DIR}
mkdir ${EVAL_DIR}

echo 'start evaluating inception-v4 on cifar100'
python ${THISPATH}/../eval_image_classifier.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=test \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${CHECKPOINT_PATH}

echo -e 'finished evaluating inception-v4 on cifar100\n\n\n'

