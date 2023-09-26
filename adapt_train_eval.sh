train="./dataset/train_fine.json"
val="./dataset/val_fine.json"
test="./dataset/test_fine.json"

train_sample="./dataset/train_sample.json"
train_g_sample="./dataset/train_generative_sample_final.json"

train_b="./dataset/train_basic.json"
val_b="./dataset/val_basic.json"
test_b="./dataset/test_basic.json"
  
train_c="./dataset/train_no_roller.json"  
train_no_ed="./dataset/atrain_basic_no_ed.json"
test_no_ed="./dataset/atest_basic_no_ed.json"
val_no_ed="./dataset/aval_basic_no_ed.json"

train_no_ed_lower="./dataset/atrain_basic_no_ed_lower.json"
test_no_ed_lower="./dataset/atest_basic_no_ed_lower.json"
val_no_ed_lower="./dataset/aval_basic_no_ed_lower.json"

train_no_anger_f4="./dataset/train_no_anger_f4.json"
test_no_anger_f4="./dataset/test_no_anger_f4.json"
val_no_anger_f4="./dataset/val_no_anger_f4.json"

# WANDB_PROJECT=adapter_emo WANDB_ENTITY=haru-emo python adapter_lora.py \
#   --model_name_or_path microsoft/deberta-v3-large --fp16 \
#   --train_file  ${train_b} \
#   --validation_file  ${val_b} \
#   --dataset_columns sentence \
#   --label_names labels \
#   --do_train --per_device_train_batch_size 8 \
#   --gradient_accumulation_steps 16 \
#   --do_eval --per_device_eval_batch_size 16 \
#   --logging_strategy steps --logging_steps 0.01 \
#   --evaluation_strategy steps --eval_steps 0.05 --save_strategy steps --save_steps 0.1 \
#   --optim adamw_torch --learning_rate 4e-5 --weight_decay 0.01 --warmup_ratio 0.02\
#   --lora_target_modules query_proj,value_proj \
#   --save_total_limit 3 \
#   --num_train_epochs 10 \
#   --max_seq_length 128 \
#   --output_dir ./exp/lora/lora_debertav_cosrestart/train/ --overwrite_output_dir \
#   --report_to wandb --run_name lora_debertav_cosrestart \
#   --load_best_model_at_end --metric_for_best_model f1


# python adapter_lora.py \
#   --lora_name_or_path ./exp/lora/lora_debertav_cosrestart/train/ --fp16 \
#   --train_file  ${train_b} \
#   --validation_file  ${test_b} \
#   --dataset_columns sentence \
#   --label_names labels \
#   --do_train False --do_eval --per_device_eval_batch_size 64 \
#   --max_seq_length 128 \
#   --report_to none --output_dir ./exp/lora/lora_debertav_cosrestart/test/

# WANDB_PROJECT=adapter_emo WANDB_ENTITY=haru-emo python adapter_lora.py \
#   --model_name_or_path microsoft/deberta-v2-xxlarge \
#   --train_file  ${train_b} \
#   --validation_file  ${val_b} \
#   --dataset_columns sentence \
#   --label_names labels \
#   --do_train --per_device_train_batch_size 8 \
#   --gradient_accumulation_steps 16 \
#   --do_eval --per_device_eval_batch_size 16 \
#   --logging_strategy steps --logging_steps 0.01 \
#   --evaluation_strategy steps --eval_steps 0.05 --save_strategy steps --save_steps 0.1 \
#   --optim adamw_torch --learning_rate 1e-4 --weight_decay 0.01 --warmup_ratio 0.02\
#   --lora_target_modules query_proj,value_proj \
#   --save_total_limit 3 \
#   --num_train_epochs 10 \
#   --max_seq_length 128 \
#   --output_dir ./exp/lora/lora_xxlarge/train/ --overwrite_output_dir \
#   --report_to wandb --run_name lora_xxlarge \
#   --load_best_model_at_end --metric_for_best_model f1

# python adapter_lora.py \
#   --lora_name_or_path ./exp/lora/lora_xxlarge/train/checkpoint-6180 --fp16 \
#   --train_file  ${train_b} \
#   --validation_file  ${test_b} \
#   --dataset_columns sentence \
#   --label_names labels \
#   --do_train False --do_eval --per_device_eval_batch_size 64 \
#   --max_seq_length 128 \
#   --report_to none --output_dir ./exp/lora/lora_xxlarge/test/

# python adapter_lora.py \
#   --lora_name_or_path ./exp/lora/lora_xxlarge/train/checkpoint-6180 --fp16 \
#   --train_file  ${train_b} \
#   --validation_file  ${val_b} \
#   --dataset_columns sentence \
#   --label_names labels \
#   --do_train False --do_eval --per_device_eval_batch_size 64 \
#   --max_seq_length 128 \
#   --report_to none --output_dir ./exp/lora/lora_xxlarge/val/



#  WANDB_PROJECT=adapter_emo WANDB_ENTITY=haru-emo python adapter_lora.py \
#   --model_name_or_path roberta-large --fp16 \
#   --train_file  ${train_no_anger_f4} \
#   --validation_file  ${val_no_anger_f4} \
#   --dataset_columns sentence \
#   --label_names labels \
#   --do_train --per_device_train_batch_size 32 --gradient_accumulation_steps 4 \
#   --do_eval --per_device_eval_batch_size 32 \
#   --logging_strategy steps --logging_steps 0.01 \
#   --evaluation_strategy steps --eval_steps 0.05 --save_strategy steps --save_steps 0.1 \
#   --optim adamw_torch --learning_rate 8e-4 --weight_decay 0.01 --warmup_ratio 0.05 \
#   --save_total_limit 2 \
#   --num_train_epochs 20 \
#   --max_seq_length 128 \
#   --output_dir .exp/lora/lora-roberta-large_no_anger_f4/ --overwrite_output_dir \
#   --report_to wandb --run_name lora-roberta-large_no_anger_f4\
#   --load_best_model_at_end --metric_for_best_model f1 \
#   --push_to_hub true \

# python adapter_lora.py \
#   --lora_name_or_path .exp/lora/lora-roberta-large_no_anger_f4/ --fp16 \
#   --train_file  ${train_no_anger_f4} \
#   --validation_file  ${test_no_anger_f4} \
#   --dataset_columns sentence \
#   --label_names labels \
#   --do_train False --do_eval --per_device_eval_batch_size 64 \
#   --max_seq_length 128 \
#   --report_to none --output_dir ./exp/lora/lora-roberta-large_no_anger_f4/test/

#  WANDB_PROJECT=adapter_emo WANDB_ENTITY=haru-emo python adapter_lora.py \
#   --model_name_or_path roberta-large --fp16 \
#   --train_file  ${train_no_ed_lower} \
#   --validation_file  ${val_no_ed_lower} \
#   --dataset_columns sentence \
#   --label_names labels \
#   --do_train --per_device_train_batch_size 32 --gradient_accumulation_steps 4 \
#   --do_eval --per_device_eval_batch_size 32 \
#   --logging_strategy steps --logging_steps 0.01 \
#   --evaluation_strategy steps --eval_steps 0.05 --save_strategy steps --save_steps 0.1 \
#   --optim adamw_torch --learning_rate 1e-3 --weight_decay 0.01 --warmup_ratio 0.05 \
#   --save_total_limit 2 \
#   --num_train_epochs 20 \
#   --max_seq_length 128 \
#   --output_dir .exp/lora/lora-roberta-large-no-ed-lower/ --overwrite_output_dir \
#   --report_to wandb --run_name lora-roberta-large-no-ed-lower \
#   --load_best_model_at_end --metric_for_best_model f1 \
#   --push_to_hub true \

python adapter_lora.py \
  --lora_name_or_path /home/annie/Desktop/haru-nlp-train-fork/src/haru_nlp_train/text-classification/.exp/lora/lora-roberta-large-no-ed --fp16 \
  --train_file  ${train_no_ed} \
  --validation_file  ${test_no_ed} \
  --dataset_columns sentence \
  --label_names labels \
  --do_train False --do_eval --per_device_eval_batch_size 64 \
  --max_seq_length 128 \
  --report_to none --output_dir ./exp/lora/lora-roberta-large-no-ed/test/


#  WANDB_PROJECT=adapter_emo WANDB_ENTITY=haru-emo python adapter_lora.py \
#   --model_name_or_path roberta-base --fp16 \
#   --train_file  ${train_b} \
#   --validation_file  ${val_b} \
#   --dataset_columns sentence \
#   --label_names labels \
#   --do_train --per_device_train_batch_size 128 \
#   --do_eval --per_device_eval_batch_size 64 \
#   --logging_strategy steps --logging_steps 0.01 \
#   --evaluation_strategy steps --eval_steps 0.05 --save_strategy steps --save_steps 0.1 \
#   --optim adamw_torch --learning_rate 1e-3 --weight_decay 0.01 --warmup_ratio 0.05 \
#   --save_total_limit 3 \
#   --num_train_epochs 10 \
#   --max_seq_length 128 \
#   --output_dir ./exp/lora/lora_base0808-2/train/ --overwrite_output_dir \
#   --report_to wandb --run_name lora_base0808-2 \
#   --load_best_model_at_end --metric_for_best_model f1

# python adapter_lora.py \
#   --lora_name_or_path ./exp/lora/lora_base0808-2/train/ --fp16 \
#   --train_file  ${train_b} \
#   --validation_file  ${val_b} \
#   --dataset_columns sentence \
#   --label_names labels \
#   --do_train False --do_eval --per_device_eval_batch_size 64 \
#   --max_seq_length 128 \
#   --report_to none --output_dir ./exp/lora/lora_base0808-2/eval/



