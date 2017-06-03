DATASET_ROOT=/home/cideep/Work/tensorflow/datasets/Link-to-datasets
datasets=( ${DATASET_ROOT}/* )
for i in ${!datasets[@]}; do
	datasets[$i]=${datasets[$i]/"${DATASET_ROOT}/"/""}
done
echo "${datasets[@]}"

