_base_ = "./base.py"

### wandb settings
file_base_dir_name = "{{fileDirname}}".split("/")[-2]
wandb_job_name = "PG_" + file_base_dir_name + '_' + '{{fileBasenameNoExtension}}'

### path & device settings
output_path_base = "{path to output}"
asset_root = "./assets"
font_root = "{path to fonts}"

### Dataset Settings
resolution = 512
train_batch_size = 32
dataloader_num_workers = 16
dataloader_pin_memory = True
dataloader_shuffle = True
dataset_cfg = dict(
    resolution=resolution,
    asset_root=asset_root,
    font_root=font_root,
    char_set="all",
    samples_per_style=5000,
    augment_color_p=0.8,
)

### Model Settings
cast_training_params = True

### Training Settings
# loss
lpips_scale = 0.1
a_scale = 10.0
# iteration
num_train_epochs = 1
max_train_steps = None
checkpointing_steps = 2000
resume_from_checkpoint = "latest"
gradient_accumulation_steps = 1

# lr
optimizer = "adamw"
lr = 1e-5
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
validation_steps = 1000
validation_cases = [
    {
        "text": "你好世界",
        "font_base": "FZRunYuYuanSongS.TTF",
        "aug_seed": -1,
    },
    {
        "text": "北京欢迎你",
        "font_base": "Muyao_Regular.TTF",
        "aug_seed": -1,
    },
    {
        "text": "你好世界",
        "font_base": "FZFengYaKaiSongS_DemiBold.TTF",
        "aug_seed": 0,
    },
    {
        "text": "北京欢迎你",
        "font_base": "zihun245hao-feiyuxingshu.ttf",
        "aug_seed": 1,
    },
]