function myoutput()
{
	# input arguments
	local DATASET_NAME=$1
	local MODEL_NAME=$2

	OUTPUT_ROOT=/home/cideep/Work/tensorflow/output-data
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
}


declare -a datasets=("cifar10" "cifar100" "VOC-2012")
declare -a models=("inception_v4" "inception_resnet_v2")

for d in "${datasets[@]}"
do
	for m in "${models[@]}"
	do
		myoutput "$d" "$m"
	done
done
