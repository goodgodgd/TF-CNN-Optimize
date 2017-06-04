PYTHON_PATH=/home/cideep/Work/tensorflow/tfenv/bin
SCRIPT=`realpath $0`
THISPATH=`dirname $SCRIPT`
CHECKPOINT_ROOT=/home/cideep/Work/tensorflow/checkpoints/Link-to-checkpoints
DATASET_ROOT=/home/cideep/Work/tensorflow/datasets/Link-to-datasets

function mytest()
{
	# input arguments
	local DATASET_NAME=$1
	local MODEL_NAME=$2
	
	DATASET_DIR=${DATASET_ROOT}/${DATASET_NAME}/tfrecord
	CHECKPOINT_DIR=${CHECKPOINT_ROOT}/${MODEL_NAME}_${DATASET_NAME}
	echo -e "checkpoint in ${CHECKPOINT_DIR}"
	if [ ! -d "${CHECKPOINT_DIR}" ]; then
		echo -e "checkpoint dir does not exits, skip this config"
		return
	fi

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


declare -a datasets=("voc2012" "cifar10" "cifar100")
declare -a models=("resnet_v2_50" "resnet_v2_101" "vgg_16" "vgg_19" "inception_resnet_v2" "inception_v4")
declare -a scopes=("resnet_v2_50" "resnet_v2_101" "vgg_16" "vgg_19" "InceptionResnetV2" "InceptionV4")

for data in "${datasets[@]}"; do
	for i in ${!models[@]}; do
		mytest "$data" "${models[i]}"
	done
done

