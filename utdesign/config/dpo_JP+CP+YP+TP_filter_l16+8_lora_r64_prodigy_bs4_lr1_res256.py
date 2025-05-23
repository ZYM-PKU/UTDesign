_base_ = "./base.py"

### wandb settings
file_base_dir_name = "{{fileDirname}}".split("/")[-2]
wandb_job_name = "PG_" + file_base_dir_name + '_' + '{{fileBasenameNoExtension}}'

### path & device settings
output_path_base = "{path to output}"
data_roots = ["{path to data1}", "{path to data2}", "..."]
dpo_data_roots = ["{path to dpo data1}", "{path to dpo data2}", "..."]
asset_root = "./assets"
std_font_path = "./assets/NotoSansSC-Regular.ttf"
expand_ratio = 0.0
max_items_per_box = 24

### Dataset Settings
resolution = 256
train_batch_size = 4
dataloader_num_workers = 4
dataloader_pin_memory = True
dataloader_shuffle = True
dataset_names = ["CollectedGlyphDPODataset"] * len(data_roots)
dataset_cfgs = [
    dict(
        data_root=data_root,
        dpo_data_root=dpo_data_root,
        asset_root=asset_root,
        std_font_path=std_font_path,
        expand_ratio=expand_ratio,
        max_items_per_box=max_items_per_box,
    )
    for data_root, dpo_data_root in zip(data_roots, dpo_data_roots)
]

### Model Settings
stage1_model_path = "{path to pretrained model}"
fusion_model_path = "{path to fusion model}"
trans_vae_dir = "{path to transparency vae}"
fuse_lora_paths = None

### Training Settings
cast_training_params = True
freeze_fusion = True
rank = 64
beta_dpo = 2000.0

# cfg drop
drop_rate = 0.1
# timestep
weighting_scheme = "none"
logit_mean = 0.0
logit_std = 1.0
mode_scale = -0.54 # 1.29

# iteration
num_train_epochs = 50
max_train_steps = None
checkpointing_steps = 1000
resume_from_checkpoint = "latest"
gradient_accumulation_steps = 1

# lr
optimizer = "prodigy"
lr = 1.0
lr_fusion = 0.0
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
validation_steps = 500
validation_case_indices = [0, 4, 8, 12]