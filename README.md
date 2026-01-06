<h2 align="center">UTDesign: A Unified Framework for Stylized Text Editing and Generation in Graphic Design Images</h2>

<p align="center">
  Yiming Zhao<sup>1</sup>, Yuanpeng Gao<sup>1</sup>, Yuxuan Luo<sup>1</sup>, Jiwei Duan<sup>2</sup>, Shisong Lin<sup>2</sup>, Longfei Xiong<sup>2</sup>, Zhouhui Lian<sup>1†</sup>
  <br>
  <sup>1</sup>Wangxuan Institute of Computer Technology, Peking University, <sup>2</sup>Kingsoft Office
  <br>
  <sup>†</sup>Corresponding author
</p>

<p align="center">
  <a href='https://arxiv.org/abs/2512.20479'><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href='https://utdesign-official.github.io/home/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
  <a href='https://huggingface.co/UTDesign/UTDesign_v1.0'><img src='https://img.shields.io/badge/Model-Huggingface-yellow?logo=huggingface&logoColor=yellow' alt='Model'></a>

<p align="center"><img src="./assets/teaser.jpg" width="100%"></p>

<span style="font-size: 12px; font-weight: 400;">UTDesign supports editing arbitrary stylized text in design images (A) as well as generating complete design images (B). On the left side, we illustrate the pipeline for the two tasks, while the right side showcases the results of UTDesign across three different applications: (1) stylized text editing, (2) conditional stylized text generation, and (3) full design image generation.</span>

<!-- Features -->
## 🌟 Features
- **Style-preserved Text Editing**: Accept arbitrary number of glyph style references for glyph style transfer with the help of a CLIP-based style encoder.
- **Accurate Bilingual Glyph Generation**: Supports accurate glyph generation (especially Chinese glyphs) with diffusion models using DINOv2-based visual conditioning.
- **RGBA Forground Output**: Supports 4-channel (RGBA) forground glyph image outputs for flexible usage based on transparency VAE decoding.


<!-- TODO List -->
## 🚧 TODO List
- [x] Release inference pipelines.
- [x] Release gradio demo.
- [x] Release training instructions.


<!-- Environment Setup -->
## 🛠️ Environment Setup
### Pull the Docker Image
```bash
docker pull zympku/diffdev:v4_release
```

### Clone the Repositary

```bash
cd /your/workdir
git clone https://github.com/ZYM-PKU/UTDesign.git
```

### Create Docker Container
```bash
sudo docker run --name utdesign --gpus=all -it --ipc=host --network=host -v /your/workdir:/workspace -w /workspace/UTDesign zympku/diffdev:v4_release /bin/bash
```

<!-- Download Model Weights -->
## ⬇️ Download Model Weights
Download the checkpoints using `Huggineface CLI`:
```
huggingface-cli download UTDesign/UTDesign_v1.0 --local-dir checkpoints
```

The downloaded checkpoints should be organized as follows:
```
checkpoints/
├── tools/
│   ├── big-lama
│   │   ├── models
|   |   |   └── best.ckpt
│   │   └── config.yaml
│   └── yolo.pt
└── utdesign_l16+8_lora64_res256/
    ├── fusion_module
    |   ├── config.json
    |   └── diffusion_pytorch_model.safetensors
    ├── trans_vae
    |   ├── config.json
    |   └── diffusion_pytorch_model.safetensors
    ├── transformer
    |   ├── config.json
    |   └── diffusion_pytorch_model.safetensors
    └── pytorch_lora_weights.safetensors
```


## ▶️ Inference Pipelines
### Layout Planner
- First of all, deploy the layout planner on gpu `{your gpu id}` of your local machine:
```
sudo docker run --gpus=all zympku/vllm:v1_release /bin/bash -c "CUDA_VISIBLE_DEVICES={your gpu id} vllm serve stage1+2_grpo1_800 --port 8000 --served-model-name vllm_layout_planner --max-model-len 10000 --trust-remote-code --limit-mm-per-prompt image=1"
```

### Stylized text editing
<p align="center"><img src="./assets/show_edit.png" width="100%"></p>

- Run the following command to try stylized text editing on given examples:
```bash
python test_edit.py
```

- To try it on your own data, first arrange your cases as in `assets/edit_cases.json` and run the following command:
```bash
python test_edit.py --anno_path /path/to/your/json
```

### Full design image generation
<p align="center"><img src="./assets/show_full_gen.png" width="100%"></p>

- Run the following command to try full design image generation on given examples:
```bash
python test_full_gen.py
```

- To try it on your own data, first arrange your cases as in `assets/full_gen_cases.json` and run the following command:
```bash
python test_full_gent.py --anno_path /path/to/your/json
```

## 🕹️ Gradio Demo
- Run the following command to create a demo page hosted on your local machine:
```
python app.py
```

## 🧪 Training Instructions
### Train the transparency VAE
- Fill in the required fields of the config file and run the following command to train the transparency VAE:
```bash
accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml \
    trans_vae/train.py trans_vae/config/sft_skip_connect_bs32_lr1e-5_res512.py
```

### Stage1: pre-train
- First, fill in the required fields of the config file and run the following command to train on gray-scale fonts:
```bash
accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml \
    pretrain/train.py pretrain/config/tfs_l16+8_prodigy_bs16_lr1_res256.py
```

- Then, fill in the required fields of the config file and run the following command to train on colored fonts with augmentation:
```bash
accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml \
    pretrain/train.py pretrain/config/tfs_l16+8_prodigy_bs16_lr1_res256_colored.py
```

- Finally, fill in the required fields of the config file and run the following command to fine-tune the model on the design image dataset:
```bash
accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml \
    utdesign/train_lora_editing.py utdesign/config/te_JP+CP+YP+TP_filter_l16+8_prodigy_bs16_lr1_res256.py
```

### Stage2: Alignment
- Fill in the required fields of the config file and run the following command to conduct feature alignment:
```bash
accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml \
    utdesign/train_fusion.py utdesign/config/tf_JP+CP+YP+TP_filter_3B_seq8_l2_prodigy_bs4_lr1.py
```

### Stage3: post-train
- First, fill in the required fields of the config file and run the following command to fine-tune the model:
```bash
accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml \
    utdesign/train_lora.py utdesign/config/sft_JP+CP+YP+TP_filter_l16+8_lora_r64_prodigy_bs4_lr1e5_res256.py
```

- Then, fill in the required fields of the config file and run the following command to conduct Diffusion-DPO:
```bash
accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml \
    utdesign/train_lora_dpo.py utdesign/config/dpo_JP+CP+YP+TP_filter_l16+8_lora_r64_prodigy_bs4_lr1_res256.py
```


## 🎉 Acknowledgement
- Datasets: We sincerely appreciate [Kingsoft](www.kingsoft.com) Corporation for providing part of the data with fine-grained annotations.
- Code & Model: Our project is built on the [diffusers](https://github.com/huggingface/diffusers) code base and we leverage the weights of [FLUX](https://github.com/black-forest-labs/flux) VAE.

### 🪬 Citation

```
@inproceedings{zhao2025utdesign,
  title={UTDesign: A Unified Framework for Stylized Text Editing and Generation in Graphic Design Images},
  author={Zhao, Yiming and Gao, Yuanpeng and Luo, Yuxuan and Duan, Jiwei and Lin, Shisong and Xiong, Longfei and Lian, Zhouhui},
  booktitle={Proceedings of the SIGGRAPH Asia 2025 Conference Papers},
  pages={1--11},
  year={2025}
}
```
