DATASET_NAME=cifar10
DATASET_DIR=/home/cideep/Work/tensorflow/datasets/cifar-10/tfrecord

MODEL_NAME=inception_v4
CHECKPOINT_PATH=/home/cideep/Work/tensorflow/checkpoints/fine-tuned/model.ckpt-1000

TRAIN_DIR=/home/cideep/Work/tensorflow/checkpoints/fine-tuned

python train_image_classifier.py \
    --eval_dir=${TRAIN_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --batch_size=32 \
    --max_num_batches=312 \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${CHECKPOINT_PATH}

