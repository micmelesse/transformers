# install deps
pip3 install tensorboard tensorboardX datasets
pip3 install ninja
# pip3 install -v --install-option="--cpp_ext" --install-option="--cuda_ext" 'git+https://github.com/ROCmSoftwarePlatform/apex.git'
sudo apt update
sudo apt install unzip -y

# install lib
pip3 install .

# download data
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1
DATA_DIR=data
mkdir -p $DATA_DIR
mv wikitext-2-raw $DATA_DIR

# single node
export TRAIN_FILE=$DATA_DIR/wikitext-2-raw/wiki.train.raw
export TEST_FILE=$DATA_DIR/wikitext-2-raw/wiki.test.raw

python3.6 -m torch.distributed.launch --nproc_per_node=8 ./examples/language-modeling/run_clm.py \
    --output_dir=output \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --overwrite_output_dir
