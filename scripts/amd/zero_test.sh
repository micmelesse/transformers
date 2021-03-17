# use bash
SCRIPT_PATH=$(realpath $0)
SCRIPT_DIR_PATH=$(dirname $SCRIPT_PATH)
ROOT_DIR=$(dirname $SCRIPT_DIR_PATH)
if [ ! "$BASH_VERSION" ]; then
    echo "Using bash to run this script $0" 1>&2
    exec bash $SCRIPT_PATH "$@"
    exit 1
fi

# copy deepsped config files
sh scripts/amd/copy_deepspeed_configs.sh

export BS=40
echo "baseline"
USE_TF=0 python -m torch.distributed.launch --nproc_per_node=2 \
    examples/seq2seq/run_translation.py \
    --model_name_or_path t5-large \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_train_samples 500 \
    --max_val_samples 500 \
    --logging_steps 1 \
    2>&1 | tee log_baseline.txt

# export BS=52
# echo "w/ --fp16"
# USE_TF=0 python -m torch.distributed.launch --nproc_per_node=2 \
#     examples/seq2seq/run_translation \
#     --model_name_or_path t5-large \
#     --do_train \
#     --do_eval \
#     --task translation_en_to_ro \
#     --dataset_name wmt16 \
#     --dataset_config_name ro-en \
#     --source_prefix "translate English to Romanian: " \
#     --output_dir /tmp/tst-translation \
#     --per_device_train_batch_size=$BS \
#     --per_device_eval_batch_size=$BS \
#     --overwrite_output_dir \
#     --predict_with_generate \
#     --max_train_samples 500 \
#     --max_val_samples 500 \
#     --logging_steps 1 \
#     --fp16 \
#     2>&1 | tee log_fp16.txt

# export BS=54
# echo "w/ --sharded_ddp"
# USE_TF=0 python -m torch.distributed.launch --nproc_per_node=2 \
#     examples/seq2seq/run_translation \
#     --model_name_or_path t5-large \
#     --do_train \
#     --do_eval \
#     --task translation_en_to_ro \
#     --dataset_name wmt16 \
#     --dataset_config_name ro-en \
#     --source_prefix "translate English to Romanian: " \
#     --output_dir /tmp/tst-translation \
#     --per_device_train_batch_size=$BS \
#     --per_device_eval_batch_size=$BS \
#     --overwrite_output_dir \
#     --predict_with_generate \
#     --max_train_samples 500 \
#     --max_val_samples 500 \
#     --logging_steps 1 \
#     --sharded_ddp \
#     2>&1 | tee log_sharded_ddp.txt

# export BS=60
# echo "w/ --sharded_ddp --fp16"
# USE_TF=0 python -m torch.distributed.launch --nproc_per_node=2 \
#     examples/seq2seq/run_translation \
#     --model_name_or_path t5-large \
#     --do_train \
#     --do_eval \
#     --task translation_en_to_ro \
#     --dataset_name wmt16 \
#     --dataset_config_name ro-en \
#     --source_prefix "translate English to Romanian: " \
#     --output_dir /tmp/tst-translation \
#     --per_device_train_batch_size=$BS \
#     --per_device_eval_batch_size=$BS \
#     --overwrite_output_dir \
#     --predict_with_generate \
#     --max_train_samples 500 \
#     --max_val_samples 500 \
#     --logging_steps 1 \
#     --sharded_ddp \
#     --fp16 \
#     2>&1 | tee log_sharded_ddp_fp16.txt

# export BS=80
# echo "w/ --deepspeed ds_config.json (stage 2 w/o cpu offloading)"
# USE_TF=0 python -m torch.distributed.launch --nproc_per_node=2 \
#     examples/seq2seq/run_translation \
#     --model_name_or_path t5-large \
#     --do_train \
#     --do_eval \
#     --task translation_en_to_ro \
#     --dataset_name wmt16 \
#     --dataset_config_name ro-en \
#     --source_prefix "translate English to Romanian: " \
#     --output_dir /tmp/tst-translation \
#     --per_device_train_batch_size=$BS \
#     --per_device_eval_batch_size=$BS \
#     --overwrite_output_dir \
#     --predict_with_generate \
#     --max_train_samples 500 \
#     --max_val_samples 500 \
#     --logging_steps 1 \
#     --deepspeed "ds_config_cpu_offload_off.json" \
#     2>&1 | tee log_sharded_ddp_fp16.txt

# export BS=86
# echo "w/ --deepspeed ds_config.json (stage 2 w/ cpu offloading)"
# USE_TF=0 python -m torch.distributed.launch --nproc_per_node=2 \
#     examples/seq2seq/run_translation \
#     --model_name_or_path t5-large \
#     --do_train \
#     --do_eval \
#     --task translation_en_to_ro \
#     --dataset_name wmt16 \
#     --dataset_config_name ro-en \
#     --source_prefix "translate English to Romanian: " \
#     --output_dir /tmp/tst-translation \
#     --per_device_train_batch_size=$BS \
#     --per_device_eval_batch_size=$BS \
#     --overwrite_output_dir \
#     --predict_with_generate \
#     --max_train_samples 500 \
#     --max_val_samples 500 \
#     --logging_steps 1 \
#     --deepspeed "ds_config_cpu_offload_on.json" \
#     2>&1 | tee log_sharded_ddp_fp16.txt
