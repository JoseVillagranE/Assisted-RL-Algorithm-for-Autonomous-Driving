#!bin/bash

GPU=$1
shift
DIR=/home/Tesis/models_logs

echo GPU = $GPU
echo Container Directory: $DIR

NV_GPU="$GPU" nvidia-docker run --rm --name carla_training --net=host -v $HOME/:$DIR -t carla-collavoi $@
