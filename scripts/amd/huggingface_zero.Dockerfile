# Select base Image
# FROM rraminen/deepspeed:DeepSpeed_Megatron-LM-GPT2_bingBERT_rocm4.0
FROM jithunnair/pytorch:rocm4.1_ubuntu18.04_py3.6_pytorch_deepspeed



# Install dependencies
RUN apt update && apt install -y \
    unzip 
RUN pip3 install regex sacremoses filelock gitpython rouge_score sacrebleu datasets fairscale

# copy repo to workspace
WORKDIR /workspace
COPY . transformers/
RUN cd transformers/ && \
    python3 -m pip install --no-cache-dir .

# set work dir
WORKDIR /workspace/transformers

# download models
RUN transformers-cli download t5-small
# RUN transformers-cli download t5-base
RUN transformers-cli download t5-large
# RUN transformers-cli download t5-3b
# RUN transformers-cli download t5-11b