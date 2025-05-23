# qwen2.5vl-3B
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train config/qwen25-vl-3B_lora_sft_stage1+2_train.yaml

# qwen2.5vl-7B
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train config/qwen25-vl-7B_lora_sft_stage1+2_train.yaml
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=4,5,6,7 llamafactory-cli train config/qwen25-vl-7B_lora_sft_stage1+2_train.yaml