# alias drun='sudo docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined'
alias drun='sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME/dockerx:/dockerx --ipc=host --shm-size=64G'
# alias drun='sudo docker run -it --privileged --device=/dev/kfd --device=/dev/dri --group-add video --ipc="host" --pid=host --shm-size=4g'
VOLUMES="-v $HOME/dockerx:/dockerx"
WORK_DIR='-w /dockerx/transformers'
# WORK_DIR='-w /root/transformers'

# IMAGE_NAME=rocm/pytorch:rocm4.0_ubuntu18.04_py3.6_pytorch_1.7.0_apex_c1e88fa
# IMAGE_NAME=rocm/tensorflow:rocm4.0.1-tf2.3-dev
# IMAGE_NAME=rocm/pytorch:rocm4.0_ubuntu18.04_py3.6_pytorch
# IMAGE_NAME=rocm/pytorch:rocm4.0.1_ubuntu18.04_py3.6_pytorch
# IMAGE_NAME=rocm/pytorch-private:rocm4.1rel_ub18_rocm_41_internal_pytorch_bnorm_patch_SSD_updated
# IMAGE_NAME=zero
# IMAGE_NAME=compute-artifactory.amd.com:5000/rocm-plus-docker/framework/compute-rocm-rel-4.1:21_ubuntu18.04_py3.6_pytorch_rocm4.1_internal_testing_169a263_30
IMAGE_NAME=rraminen/deepspeed:DeepSpeed_Megatron-LM-GPT2_bingBERT_rocm4.0

CONTAINER_ID=$(drun -d $WORK_DIR $VOLUMES $IMAGE_NAME)
echo "CONTAINER_ID: $CONTAINER_ID"
# docker cp scripts/amd $CONTAINER_ID:/root/transformers/scripts
docker attach $CONTAINER_ID
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID
