# Select base Image
FROM rraminen/deepspeed:DeepSpeed_Megatron-LM-GPT2_bingBERT_rocm4.0

# Install dependencies
RUN apt update && apt install -y \
    unzip 
RUN pip3 install regex sacremoses filelock gitpython rouge_score sacrebleu datasets fairscale

# install transformers
RUN cd /root &&\
    git clone https://github.com/huggingface/transformers &&\
    cd transformers &&\
    pip install -e .

# set work dir
WORKDIR /root/transformers






