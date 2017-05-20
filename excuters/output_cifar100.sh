# changing lines
DATASET_NAME=cifar100
MODEL_NAME=inception_v4
OUTPUT_ROOT=/home/cideep/Work/tensorflow/output-data

# below lines are fixed
PYTHON_PATH=/home/cideep/Work/tensorflow/tfenv/bin
SCRIPT=`realpath $0`
THISPATH=`dirname $SCRIPT`

CHECKPOINT_PATH=/home/cideep/Work/tensorflow/checkpoints/my-fine-tuned/${MODEL_NAME}_${DATASET_NAME}/model.ckpt-50000
OUTPUT_DIR=${OUTPUT_ROOT}/${MODEL_NAME}_${DATASET_NAME}

SPLIT_NAME=test
echo "start outputing probabilities from ${MODEL_NAME} on ${SPLIT_NAME} of ${DATASET_NAME}"
${PYTHON_PATH}/python ${THISPATH}/../run_inference.py \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_name=${DATASET_NAME} \
    --split_name=${SPLIT_NAME} \
    --output_dir=${OUTPUT_DIR} \
    --batch_size=32

SPLIT_NAME=validation
echo "start outputing probabilities from ${MODEL_NAME} on ${SPLIT_NAME} of ${DATASET_NAME}"
${PYTHON_PATH}/python ${THISPATH}/../run_inference.py \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_name=${DATASET_NAME} \
    --split_name=${SPLIT_NAME} \
    --output_dir=${OUTPUT_DIR} \
    --batch_size=32

echo -e "finished outputing probabilities from ${MODEL_NAME} on ${DATASET_NAME}\n\n\n"

