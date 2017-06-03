import subprocess
import os

# changing lines
DATASET_NAME = "cifar10"
MODEL_NAME = "inception_v4"

# below lines are fixed
PYTHON_PATH = "/home/cideep/Work/tensorflow/tfenv/bin"

DATASET_DIR = "/home/cideep/Work/tensorflow/datasets/%s/tfrecord" % (DATASET_NAME)
CHECKPOINT_DIR = "/home/cideep/Work/tensorflow/checkpoints/my-fine-tuned/%s_%s" \
  % (MODEL_NAME, DATASET_NAME)
print("checkpoint in", CHECKPOINT_DIR)

EVAL_DIR="%s/result" % (DATASET_DIR)
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


print subprocess.Popen("echo Hello World", shell=True, stdout=subprocess.PIPE).stdout.read()
