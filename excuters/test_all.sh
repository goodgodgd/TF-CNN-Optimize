CHECKPOINT_ROOT=/home/cideep/Work/tensorflow/checkpoints/Link-to-checkpoints/my-fine-tuned
DATASET_ROOT=/home/cideep/Work/tensorflow/datasets/Link-to-datasets

function mytest()
{
	# input arguments
	local DATASET_NAME=$1
	local MODEL_NAME=$2
	
	PYTHON_PATH=/home/cideep/Work/tensorflow/tfenv/bin
	SCRIPT=`realpath $0`
	THISPATH=`dirname $SCRIPT`

	DATASET_DIR=${DATASET_ROOT}/${DATASET_NAME}/tfrecord
	CHECKPOINT_DIR=${CHECKPOINT_ROOT}/${MODEL_NAME}_${DATASET_NAME}
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
}


declare -a datasets=("cifar10" "cifar100" "VOC-2012")
declare -a models=("inception_v4" "inception_resnet_v2")

for d in "${datasets[@]}"
do
	for m in "${models[@]}"
	do
		mytest "$d" "$m"
	done
done

