_base_ = "./base.py"

### wandb settings
file_base_dir_name = "{{fileDirname}}".split("/")[-2]
wandb_job_name = "PG_" + file_base_dir_name + '_' + '{{fileBasenameNoExtension}}'

### path & device settings
pretrained_ckpt_dir = "{path to pretrained ckpt}"
output_path_base = "{path to output}"
asset_root = "./assets"
font_root = "{path to fonts}"
std_font_path = "./assets/NotoSansSC-Regular.ttf"

### Dataset Settings
resolution = 256
train_batch_size = 16
dataloader_num_workers = 16
dataloader_pin_memory = True
dataloader_shuffle = True
dataset_cfg = dict(
    resolution=resolution,
    asset_root=asset_root,
    font_root=font_root,
    std_font_path=std_font_path,
    char_set="all",
    samples_per_style=1000,
    interval_c_refs=[5,10], 
    interval_s_refs=[2,10],
    augment_color_texture=True,
    disturb_style=True,
)


### Model Settings
cast_training_params = True
trans_vae_dir = "{path to transparency vae}"
fusion_attn_heads = 16
fusion_dim_head = 256
clip_model_name = "openai/clip-vit-large-patch14"
dino_model_name = "facebook/dinov2-large"
transformer_config = dict(
    in_channels = 64,
    num_layers = 16,
    num_single_layers = 8,
    attention_head_dim = 128,
    num_attention_heads = 8,
    joint_attention_dim = 1024, # context dimension
    pooled_projection_dim = 1024,
    axes_dims_rope = (16, 56, 56), # 3d rope
)

### Training Settings
# cfg drop
drop_rate = 0.1
# timestep
weighting_scheme = "none"
logit_mean = 0.0
logit_std = 1.0
mode_scale = 1.29

# iteration
num_train_epochs = None
max_train_steps = 60_000
checkpointing_steps = 5000
resume_from_checkpoint = "latest"
gradient_accumulation_steps = 1

# lr
optimizer = "prodigy"
lr = 1.0
lr_fusion = 1.0
scale_lr = False
lr_scheduler = "constant"
lr_warmup_steps = 0
lr_num_cycles = 1
lr_power = 1.0

# optim
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 1e-3
adam_epsilon = 1e-8
prodigy_beta3 = None
prodigy_decouple = True
prodigy_use_bias_correction = True
prodigy_safeguard_warmup = True
max_grad_norm = 1.0

# logging
tracker_task_name = file_base_dir_name + "_" + "{{fileBasenameNoExtension}}"
output_dir = output_path_base + file_base_dir_name + "_" + "{{fileBasenameNoExtension}}"

### Validation Settings
validation_steps = 2000
validation_cases = [
    {
        "text": "你好世界",
        "style_text": "这些是参考字形",
        "font_base": "FZRunYuYuanSongS.TTF",
        "aug_seed": 0,
    },
    {
        "text": "北京欢迎你",
        "style_text": "这些是参考字形",
        "font_base": "Muyao_Regular.TTF",
        "aug_seed": 1,
    },
]