alias drun='sudo docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME/dockerx:/dockerx -v /data:/data'

WORK_DIR='/dockerx/transformers'
drun -w $WORK_DIR rocm/pytorch:rocm4.0_ubuntu18.04_py3.6_pytorch_1.7.0