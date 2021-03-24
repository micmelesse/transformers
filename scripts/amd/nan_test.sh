export BS=2
echo "w/ --fp16"
cp scripts/amd/tracer.py src/transformers

#export HIP_VISIBLE_DEVICES=2
export CUDA_VISIBLE_DEVICES=2
export USE_TRACER=0
export USE_TF=0
rm -rf $HOSTNAME

# python examples/seq2seq/run_translation.py \
python -m torch.distributed.launch --nproc_per_node=1 examples/seq2seq/run_translation.py \
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
    --fp16 \
    --fp16_backend "apex" \
    # --config_name "scripts/amd/t5_large_config_no_dropout.json" \
    2>&1 | tee log_fp16.txt
