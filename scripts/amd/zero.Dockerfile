# Select base Image
# FROM rocm/pytorch:rocm4.0.1_ubuntu18.04_py3.6_pytorch
FROM compute-artifactory.amd.com:5000/rocm-plus-docker/framework/compute-rocm-rel-4.1:21_ubuntu18.04_py3.6_pytorch_rocm4.1_internal_testing_169a263_30

# Install dependencies
RUN apt update && apt install -y \
    unzip 
RUN pip3 install regex sacremoses filelock gitpython rouge_score sacrebleu deepspeed fairscale

# install transformers
RUN cd /root &&\
    git clone https://github.com/huggingface/transformers &&\
    cd transformers &&\
    git checkout 96897a353564e45141480a0260ad1314716bace7 &&\
    pip install -e .

# set work dir
WORKDIR /root/transformers

# Download data
RUN cd examples/seq2seq &&\
    wget https://cdn-datasets.huggingface.co/translation/wmt_en_ro.tar.gz &&\
    tar -xzvf wmt_en_ro.tar.gz 





