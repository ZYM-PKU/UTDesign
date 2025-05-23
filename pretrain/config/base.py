### Model Settings
pretrained_model_name_or_path = "black-forest-labs/FLUX.1-dev"
revision = None
variant = None
cache_dir = "{path to cache}"
cache_dir_clip = "{path to cache}"

### Training Settings
seed = 42
report_to = "wandb"
tracker_project_name = "UTDesign"
wandb_entity = "my_dev"
wandb_job_name = "YOU_FORGET_TO_SET"
logging_dir = "logs"
max_train_steps = None
checkpoints_total_limit = 10

# gpu
allow_tf32 = True
gradient_checkpointing = True
mixed_precision = "bf16"

### Validation Settings
validation_steps = 2000