alias drun='sudo docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined'

VOLUMES="-v $HOME/dockerx:/dockerx"
# WORK_DIR='-w /root/transformers'


# IMAGE_NAME=rocm/pytorch:rocm4.0_ubuntu18.04_py3.6_pytorch_1.7.0_apex_c1e88fa
# IMAGE_NAME=rocm/tensorflow:rocm4.0.1-tf2.3-dev
# IMAGE_NAME=rocm/pytorch:rocm4.0_ubuntu18.04_py3.6_pytorch
# IMAGE_NAME=rocm/pytorch-private:rocm4.1rel_ub18_rocm_41_internal_pytorch_bnorm_patch_SSD_updated
IMAGE_NAME=zero

CONTAINER_NAME=${IMAGE_NAME}_container

drun -d --name $CONTAINER_NAME $WORK_DIR $VOLUMES $IMAGE_NAME
# docker cp scripts/amd $CONTAINER_NAME:/root/transformers/scripts
docker attach $CONTAINER_NAME