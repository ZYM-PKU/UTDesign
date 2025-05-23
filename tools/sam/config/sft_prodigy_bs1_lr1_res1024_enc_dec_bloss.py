_base_ = "./base.py"

### wandb settings
file_base_dir_name = "{{fileDirname}}".split("/")[-2]
wandb_job_name = "PG_" + file_base_dir_name + '_' + '{{fileBasenameNoExtension}}'

### path & device settings
output_path_base = "{path to output}"
data_root = "{path to data}"
asset_root = "./assets"
font_root = "{path to fonts}"

### Dataset Settings
train_batch_size = 1
dataloader_num_workers = 16
dataloader_pin_memory = True
dataloader_shuffle = True
dataset_cfg = dict(
    data_root=data_root,
    asset_root=asset_root,
    font_root=font_root,
    char_set="all",
    expand_ratio=0.2,
)

### Model Settings
pretrained_model_name_or_path = "facebook/sam-vit-large"
cast_training_params = True
train_encoder = True
num_train_layers = 16
lamda_focal = 0.01
lamda_dice = 0.1
focal_alpha = 0.75
focal_gamma = 2.0

### Training Settings
# iteration
num_train_epochs = 2
max_train_steps = None
checkpointing_steps = 10000
resume_from_checkpoint = "latest"
gradient_accumulation_steps = 1

# lr
optimizer = "prodigy"
lr = 1.0
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
max_grad_norm = 0.1

# logging
tracker_task_name = file_base_dir_name + "_" + "{{fileBasenameNoExtension}}"
output_dir = output_path_base + file_base_dir_name + "_" + "{{fileBasenameNoExtension}}"

### Validation Settings
validation_steps = 10000
num_validation_cases = 4