alias drun='sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --shm-size=64G'

VOLUMES="-v $HOME/dockerx:/dockerx"
WORK_DIR='-w /dockerx/transformers'
# WORK_DIR='-w /workspace/transformers'

# IMAGE_NAME=rocm/pytorch:rocm4.0_ubuntu18.04_py3.6_pytorch_1.7.0_apex_c1e88fa
# IMAGE_NAME=rocm/tensorflow:rocm4.0.1-tf2.3-dev
# IMAGE_NAME=rocm/pytorch:rocm4.0.1_ubuntu18.04_py3.6_pytorch
# IMAGE_NAME=rocm/pytorch-private:rocm4.1rel_ub18_rocm_41_internal_pytorch_bnorm_patch_SSD_updated
# IMAGE_NAME=rraminen/deepspeed:DeepSpeed_Megatron-LM-GPT2_bingBERT_rocm4.0
IMAGE_NAME=huggingface_zero

CONTAINER_ID=$(drun -d $WORK_DIR $VOLUMES $IMAGE_NAME)
echo "CONTAINER_ID: $CONTAINER_ID"
docker cp . $CONTAINER_ID:/workspace/transformers
docker attach $CONTAINER_ID
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID