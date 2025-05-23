accelerate launch --config_file accelerate_cfg/1m4g_fp16_g4567.yaml --main_process_port=29000 \
    tools/sam/train.py tools/sam/config/sft_adamw_bs1_lr1e-5_res1024_dec.py

accelerate launch --config_file accelerate_cfg/1m4g_fp16_g4567.yaml --main_process_port=29000 \
    tools/sam/train.py tools/sam/config/sft_adamw_bs1_lr1e-5_res1024_enc_dec.py

accelerate launch --config_file accelerate_cfg/1m4g_fp16_g4567.yaml --main_process_port=29000 \
    tools/sam/train.py tools/sam/config/sft_prodigy_bs1_lr1_res1024_enc_dec.py

accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml --main_process_port=29000 \
    tools/sam/train.py tools/sam/config/sft_prodigy_bs1_lr1_res1024_enc_dec_bloss.py