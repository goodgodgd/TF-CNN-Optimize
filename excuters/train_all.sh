PYTHON_PATH=/home/cideep/Work/tensorflow/tfenv/bin
SCRIPT=`realpath $0`
THISPATH=`dirname $SCRIPT`
CHECKPOINT_ROOT=/home/cideep/Work/tensorflow/checkpoints/Link-to-checkpoints
DATASET_ROOT=/home/cideep/Work/tensorflow/datasets/Link-to-datasets

function mytrain()
{
	# input arguments
	local DATASET_NAME=$1
	local MODEL_NAME=$2
	local SCOPE_VAR=$3

	DATASET_DIR=${DATASET_ROOT}/${DATASET_NAME}/tfrecord
	CHECKPOINT_PATH=${CHECKPOINT_ROOT}/${MODEL_NAME}.ckpt
	TRAIN_LOG_DIR=${CHECKPOINT_ROOT}/${MODEL_NAME}_${DATASET_NAME}
	
	if [ -d "${TRAIN_LOG_DIR}" ]; then
		echo -e "trained already, skip training ${TRAIN_LOG_DIR}"
		return
	fi
	
	echo -e "make ${TRAIN_LOG_DIR}"
	mkdir ${TRAIN_LOG_DIR}

	if [[ ${MODEL_NAME} == *"inception"* ]]; then
		CHECKPOINT_EXCLUDE="${SCOPE_VAR}/Logits,${SCOPE_VAR}/AuxLogits"
	elif [[ ${MODEL_NAME} == *"vgg"* ]]; then
		CHECKPOINT_EXCLUDE="${SCOPE_VAR}/fc8"
	elif [[ ${MODEL_NAME} == *"resnet_v2_"* ]]; then
		CHECKPOINT_EXCLUDE="${SCOPE_VAR}/logits"
	fi

	echo -e "start retraining ${MODEL_NAME} on ${DATASET_NAME}"
	${PYTHON_PATH}/python ${THISPATH}/../train_image_classifier.py \
		--train_dir=${TRAIN_LOG_DIR} \
		--optimizer=rmsprop \
		--dataset_name=${DATASET_NAME} \
		--dataset_split_name=train \
		--dataset_dir=${DATASET_DIR} \
		--batch_size=32 \
		--learning_rate=0.01 \
		--learning_rate_decay_factor=0.8 \
		--max_number_of_steps=35000 \
		--model_name=${MODEL_NAME} \
		--checkpoint_path=${CHECKPOINT_PATH} \
		--checkpoint_exclude_scopes=${CHECKPOINT_EXCLUDE}

	echo -e "finished retraining ${MODEL_NAME} on ${DATASET_NAME}"
}

declare -a datasets=("voc2012" "cifar10" "cifar100")
declare -a models=("resnet_v2_50" "resnet_v2_101" "vgg_16" "vgg_19" "inception_resnet_v2" "inception_v4")
declare -a scopes=("resnet_v2_50" "resnet_v2_101" "vgg_16" "vgg_19" "InceptionResnetV2" "InceptionV4")

for data in "${datasets[@]}"; do
	for i in ${!models[@]}; do
		mytrain "$data" "${models[i]}" "${scopes[i]}"
	done
done

