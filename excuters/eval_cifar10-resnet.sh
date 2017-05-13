SCRIPT=`realpath $0`
THISPATH=`dirname $SCRIPT`

DATASET_NAME=cifar10
DATASET_DIR=/home/cideep/Work/tensorflow/datasets/cifar-10/tfrecord

MODEL_NAME=inception_resnet_v2
CHECKPOINT_PATH=/home/cideep/Work/tensorflow/checkpoints/my-fine-tuned/inc-resnet-v2-cifar10/model.ckpt-50000

echo 'start retraining inception_resnet_v2 on cifar10'
python ${THISPATH}/../run_inference.py \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --num_classes=10

echo -e 'finished retraining inception_resnet_v2 on cifar10\n\n\n'

