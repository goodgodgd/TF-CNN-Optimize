PYTHON_PATH=/home/cideep/Work/tensorflow/tfenv/bin
SCRIPT=`realpath $0`
THISPATH=`dirname $SCRIPT`
CHECKPOINT_ROOT=/home/cideep/Work/tensorflow/checkpoints/Link-to-checkpoints
DATASET_ROOT=/home/cideep/Work/tensorflow/datasets/Link-to-datasets
OUTPUT_ROOT=/home/cideep/Work/tensorflow/output-data

function myoutput()
{
	# input arguments
	local DATASET_NAME=$1
	local MODEL_NAME=$2
	local SCOPE_VAR=$3

	DATASET_DIR=${DATASET_ROOT}/${DATASET_NAME}/tfrecord
	CHECKPOINT_DIR=${CHECKPOINT_ROOT}/${MODEL_NAME}_${DATASET_NAME}
	OUTPUT_DIR=${OUTPUT_ROOT}/${MODEL_NAME}_${DATASET_NAME}
	
	if [ ! -d "${CHECKPOINT_DIR}" ]; then
		echo -e "checkpoint is not ready, skip outputing ${OUTPUT_DIR}"
		return
	fi
	if [ -d "${OUTPUT_DIR}" ]; then
		echo -e "output computed already, skip outputing ${OUTPUT_DIR}"
		return
	fi

	SPLIT_NAME=test
	echo "start outputing probabilities from ${MODEL_NAME} on ${SPLIT_NAME} of ${DATASET_NAME}"
	${PYTHON_PATH}/python ${THISPATH}/../run_inference.py \
		--model=${MODEL_NAME} \
		--scope=${SCOPE_VAR} \
		--dataset_name=${DATASET_NAME} \
		--dataset_dir=${DATASET_DIR} \
		--checkpoint_dir=${CHECKPOINT_DIR} \
		--split=${SPLIT_NAME} \
		--output_dir=${OUTPUT_DIR} \
		--batch_size=32

	SPLIT_NAME=validation
	echo "start outputing probabilities from ${MODEL_NAME} on ${SPLIT_NAME} of ${DATASET_NAME}"
	${PYTHON_PATH}/python ${THISPATH}/../run_inference.py \
		--model=${MODEL_NAME} \
		--scope=${SCOPE_VAR} \
		--dataset_name=${DATASET_NAME} \
		--dataset_dir=${DATASET_DIR} \
		--checkpoint_dir=${CHECKPOINT_DIR} \
		--split=${SPLIT_NAME} \
		--output_dir=${OUTPUT_DIR} \
		--batch_size=32

	echo -e "finished outputing probabilities from ${MODEL_NAME} on ${DATASET_NAME}\n\n\n"
}


declare -a datasets=("voc2012" "cifar10" "cifar100")
declare -a models=("resnet_v2_50" "resnet_v2_101" "vgg_16" "vgg_19" "inception_resnet_v2" "inception_v4")
declare -a scopes=("resnet_v2_50" "resnet_v2_101" "vgg_16" "vgg_19" "InceptionResnetV2" "InceptionV4")

for data in "${datasets[@]}"; do
	for i in ${!models[@]}; do
		myoutput "$data" "${models[i]}" "${scopes[i]}"
	done
done

