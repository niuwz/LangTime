ckpt=""
llm_path=""
data_path=""
model_name=lt
mkdir -p logs/pretrain/
mkdir -p logs/runtime_logs/
task_id=single_ETTh1
running_out=logs/runtime_logs/langtime-$task_id.out
running_log=logs/pretrain/$task_id".log"

deepspeed --master_port 12345 \
    run_pt.py \
    --task_id $task_id \
    --is_training 1 \
    --checkpoints $ckpt \
    --data_dir $data_path \
    --backbone_path $llm_path \
    --backbone qwen2 \
    --model_init random \
    --model $model_name \
    --save_name langtime \
    --batch_size 24 \
    --split :80 \
    --features M \
    --patience 3 \
    --train_epochs 2 \
    --training_mode full \
    --loss_alpha 0.7 0.5 \
    --loss_alpha_type reduce \
    --warmup_rate 0.03 \
    --loss Huber \
    --huber_delta 0.4 \
    --initial_lr 1e-4 \
    --lr_decay 0.01 \
    --domain ETTh1 \
    --seq_len 96 \
    --pretrain_seq_lens 96 288 480 672 \
    --label_len 48 \
    --pred_len 96 \
    --single_pred_len 96 \
    --e_layers 4 \
    --d_model 512 \
    --d_ff 2048 \
    --output_attention \
    --n_heads 4 \
    --ts_enc patch \
    --patch_size 24 \
    --enc_mask fix:0.4 \
    --num_kv_heads 2 \
    --adapter_type linear \
    --num_workers 0 \
    --log_file $running_log \
    --deepspeed_config configs/deepspeed/ds_z2_config.json 2>&1 | tee $running_out
