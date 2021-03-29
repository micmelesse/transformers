alias drun='sudo docker run -it --network=host --runtime=nvidia --ipc=host'

# WORK_DIR=' -w /workpace/transformers'

drun $WORK_DIR huggingface_zero_nv:latest
