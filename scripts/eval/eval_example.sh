model_name=lt_eval
mkdir -p logs/eval/
mkdir -p logs/runtime_logs/
task_id=eval
running_out=logs/runtime_logs/eval_lt.out
running_log=logs/eval/$task_id".log"
ckpt=""
llm_path="" # Path to the large language model (LLM) backbone.
eval_model="" # Specific model to be loaded for evaluation.

# Select datasets for evaluation. Any number of datasets can be specified.
# data_name="ETTh1-eval"
data_name="ETTh1-eval ETTm1-eval ETTh2-eval ETTm2-eval Electricity-eval Weather-eval Exchange-eval Illness-eval Traffic-eval"

in_len=96 # Input sequence length for the model.
deepspeed --master_port 12345 --include localhost:0,1 \
    run_eval.py \
    --task_id $task_id \
    --checkpoints $ckpt \
    --data_dir ./datasets \
    --backbone_path $llm_path \
    --model_init $eval_model \
    --model $model_name \
    --eval_batch_size 128 \
    --domain $data_name \
    --seq_len $in_len \
    --tasks_lens 96 192 336 720 \
    --label_len 48 \
    --num_workers 0 \
    --plot_img \
    --log_file $running_log \
    --deepspeed_config configs/deepspeed/ds_z2_config.json 2>&1 | tee $running_out