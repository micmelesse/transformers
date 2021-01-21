cd examples/seq2seq

rm -r output_dir

export BS=30
echo "baseline"
PYTHONPATH=../../src USE_TF=0 python -m torch.distributed.launch \
    --nproc_per_node=2 ./finetune_trainer.py --model_name_or_path t5-large --output_dir output_dir \
    --adam_eps 1e-06 --data_dir wmt_en_ro --do_eval --do_train --evaluation_strategy=steps --freeze_embeds \
    --label_smoothing 0.1 --learning_rate 3e-5 --logging_first_step --logging_steps 1000 --max_source_length 128 \
    --max_target_length 128 --num_train_epochs 1 --overwrite_output_dir --per_device_eval_batch_size $BS \
    --per_device_train_batch_size $BS --predict_with_generate --eval_steps 25000 --sortish_sampler \
    --task translation_en_to_ro --test_max_target_length 128 --val_max_target_length 128 --warmup_steps 500 \
    --n_train 2000 --n_val 500

# export BS=46
# echo "w/ --fp16"
# PYTHONPATH=../../src USE_TF=0 python -m torch.distributed.launch \
#     --nproc_per_node=2 ./finetune_trainer.py --model_name_or_path t5-large --output_dir output_dir \
#     --adam_eps 1e-06 --data_dir wmt_en_ro --do_eval --do_train --evaluation_strategy=steps --freeze_embeds \
#     --label_smoothing 0.1 --learning_rate 3e-5 --logging_first_step --logging_steps 1000 --max_source_length 128 \
#     --max_target_length 128 --num_train_epochs 1 --overwrite_output_dir --per_device_eval_batch_size $BS \
#     --per_device_train_batch_size $BS --predict_with_generate --eval_steps 25000 --sortish_sampler \
#     --task translation_en_to_ro --test_max_target_length 128 --val_max_target_length 128 --warmup_steps 500 \
#     --n_train 2000 --n_val 500 --fp16

# export BS=50
# echo "w/ --sharded_ddp"
# PYTHONPATH=../../src USE_TF=0 python -m torch.distributed.launch \
#     --nproc_per_node=2 ./finetune_trainer.py --model_name_or_path t5-large --output_dir output_dir \
#     --adam_eps 1e-06 --data_dir wmt_en_ro --do_eval --do_train --evaluation_strategy=steps --freeze_embeds \
#     --label_smoothing 0.1 --learning_rate 3e-5 --logging_first_step --logging_steps 1000 --max_source_length 128 \
#     --max_target_length 128 --num_train_epochs 1 --overwrite_output_dir --per_device_eval_batch_size $BS \
#     --per_device_train_batch_size $BS --predict_with_generate --eval_steps 25000 --sortish_sampler \
#     --task translation_en_to_ro --test_max_target_length 128 --val_max_target_length 128 --warmup_steps 500 \
#     --n_train 2000 --n_val 500 --sharded_ddp

# export BS=56
# echo "w/ --sharded_ddp --fp16"
# PYTHONPATH=../../src USE_TF=0 python -m torch.distributed.launch \
#     --nproc_per_node=2 ./finetune_trainer.py --model_name_or_path t5-large --output_dir output_dir \
#     --adam_eps 1e-06 --data_dir wmt_en_ro --do_eval --do_train --evaluation_strategy=steps --freeze_embeds \
#     --label_smoothing 0.1 --learning_rate 3e-5 --logging_first_step --logging_steps 1000 --max_source_length 128 \
#     --max_target_length 128 --num_train_epochs 1 --overwrite_output_dir --per_device_eval_batch_size $BS \
#     --per_device_train_batch_size $BS --predict_with_generate --eval_steps 25000 --sortish_sampler \
#     --task translation_en_to_ro --test_max_target_length 128 --val_max_target_length 128 --warmup_steps 500 \
#     --n_train 2000 --n_val 500 --sharded_ddp --fp16

# export BS=76
# echo "w/ --deepspeed ds_config.json (stage 2 w/o cpu offloading)"
# PYTHONPATH=../../src USE_TF=0 python -m torch.distributed.launch \
#     --nproc_per_node=2 ./finetune_trainer.py --model_name_or_path t5-large --output_dir output_dir \
#     --adam_eps 1e-06 --data_dir wmt_en_ro --do_eval --do_train --evaluation_strategy=steps --freeze_embeds \
#     --label_smoothing 0.1 --learning_rate 3e-5 --logging_first_step --logging_steps 1000 --max_source_length 128 \
#     --max_target_length 128 --num_train_epochs 1 --overwrite_output_dir --per_device_eval_batch_size $BS \
#     --per_device_train_batch_size $BS --predict_with_generate --eval_steps 25000 --sortish_sampler \
#     --task translation_en_to_ro --test_max_target_length 128 --val_max_target_length 128 --warmup_steps 500 \
#     --n_train 2000 --n_val 500 --deepspeed "ds_config_cpu_offload_off.json"

# export BS=80
# echo "w/ --deepspeed ds_config.json (stage 2 w/ cpu offloading)"
# PYTHONPATH=../../src USE_TF=0 python -m torch.distributed.launch \
#     --nproc_per_node=2 ./finetune_trainer.py --model_name_or_path t5-large --output_dir output_dir \
#     --adam_eps 1e-06 --data_dir wmt_en_ro --do_eval --do_train --evaluation_strategy=steps --freeze_embeds \
#     --label_smoothing 0.1 --learning_rate 3e-5 --logging_first_step --logging_steps 1000 --max_source_length 128 \
#     --max_target_length 128 --num_train_epochs 1 --overwrite_output_dir --per_device_eval_batch_size $BS \
#     --per_device_train_batch_size $BS --predict_with_generate --eval_steps 25000 --sortish_sampler \
#     --task translation_en_to_ro --test_max_target_length 128 --val_max_target_length 128 --warmup_steps 500 \
#     --n_train 2000 --n_val 500 --deepspeed "ds_config_cpu_offload_on.json"
