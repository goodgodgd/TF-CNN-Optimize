# changing lines
DATASET_NAME=cifar10
MODEL_NAME=inception_v4

# below lines are fixed
PYTHON_PATH=/home/cideep/Work/tensorflow/tfenv/bin
SCRIPT=`realpath $0`
THISPATH=`dirname $SCRIPT`

DATASET_DIR=/home/cideep/Work/tensorflow/datasets/${DATASET_NAME}/tfrecord
CHECKPOINT_DIR=/home/cideep/Work/tensorflow/checkpoints/my-fine-tuned/${MODEL_NAME}_${DATASET_NAME}
echo -e "checkpoint in ${CHECKPOINT_DIR}"

EVAL_DIR=${CHECKPOINT_DIR}/result
rm -r ${EVAL_DIR}
mkdir ${EVAL_DIR}

echo -e "start evaluating ${MODEL_NAME} on ${DATASET_NAME}"
${PYTHON_PATH}/python ${THISPATH}/../eval_image_classifier.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=test \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${MODEL_NAME} \
    --checkpoint_dir=${CHECKPOINT_DIR}

echo -e "finished evaluating ${MODEL_NAME} on ${DATASET_NAME}\n\n\n"

