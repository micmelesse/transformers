alias drun='sudo docker run -it --network=host --runtime=nvidia --ipc=host -v $HOME/dockerx:/dockerx -v /data:/data'

WORK_DIR='/dockerx/transformers'

drun -w $WORK_DIR nvcr.io/nvidia/pytorch:20.08-py3