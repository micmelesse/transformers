alias drun='sudo docker run -it --network=host --runtime=nvidia --ipc=host'

VOLUMES="-v $HOME/dockerx:/dockerx"
WORK_DIR='-w /dockerx/transformers'

IMAGE_NAME=nvcr.io/nvidia/pytorch:20.08-py3

CONTAINER_ID=$(drun -d $WORK_DIR $VOLUMES $IMAGE_NAME)
echo "CONTAINER_ID: $CONTAINER_ID"
docker cp . $CONTAINER_ID:/workspace/transformers
docker attach $CONTAINER_ID
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID