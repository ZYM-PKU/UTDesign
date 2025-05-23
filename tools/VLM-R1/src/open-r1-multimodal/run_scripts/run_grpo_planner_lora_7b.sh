export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_NAME="Qwen25-VL-7B-Instruct"
RUN_NAME="$MODEL_NAME-GRPO-PLANNER-lora"
export LOG_PATH="logs/debug_log_$RUN_NAME.txt"
MODEL_PATH="{sft model path}"

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open-r1-multimodal/src/open_r1/grpo_layout_planner.py \
    --deepspeed src/open-r1-multimodal/local_scripts/zero2.json \
    --output_dir ../work_dirs/vlm-r1/$MODEL_NAME/lora \
    --model_name_or_path $MODEL_PATH \
    --cache_dir {cache_dir} \
    --data_path ../llamafact/data/stage1+2_train.json \
    --max_completion_length 256 \
    --num_iterations 1 \
    --num_generations 4 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --use_peft true \
    --beta 0.0 \
    --lora_r 64 \
    --lora_dropout 0.0 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true