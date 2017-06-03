function mytrain()
{
	# input arguments
	local DATASET_NAME=$1
	local MODEL_NAME=$2
	local SCOPE_VAR=$3
	
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
}

declare -a datasets=("cifar10" "cifar100" "VOC-2012")
declare -a models=("inception_v4" "inception_resnet_v2")
declare -a scopes=("InceptionV4" "InceptionResnetV2")

for data in "${datasets[@]}"
do
	for i in {0..1}
	do
		mytrain "$data" "${models[i]}" "${scopes[i]}"
	done
done

