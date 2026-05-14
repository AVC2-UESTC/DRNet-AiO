# DRNet: All-in-One Image Restoration via Prior-Guided Dynamic Reparameterization

*Ao Li, Xiaoning Liu, Sheng Li, Yapeng Du, Zhen Long, Lei Luo, Le Zhang, and Ce Zhu*

**IEEE Transactions on Multimedia (TMM), 2026**

[![arXiv](https://img.shields.io/badge/arXiv-2605.08627-b31b1b.svg)](https://arxiv.org/abs/2605.08627)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

DRNet is a novel all-in-one image restoration framework that handles multiple degradation types within a single unified model via **Prior-Guided Dynamic Reparameterization**. It efficiently addresses:

- Denoising (σ=15, 25, 50)
- Dehazing
- Deraining
- Deblurring
- Low-light Enhancement

---

## 🔧 Requirements

### Environment Setup

```bash
# Create conda environment
conda create -n drnet python=3.8
conda activate drnet

# Install dependencies
pip install -r requirements.txt
```

## 📦 Pretrained Weights
Pretrained weights are available for download:

🔗 [Download Pretrained Weights](https://drive.google.com/drive/folders/1DWWfOpdtf4f9Coar3Tn_Y7DJRZBEMOI6?usp=sharing)

Place the weights in the ckpt/ directory:
```bash
mkdir -p ckpt/5tasks ckpt/3tasks
# Download and move pretrained weights to corresponding directories
```
## 📂 Dataset Preparation
Organize your training data as follows:
```text
data/
├── Denoise/       # Denoising images (noisy/clean pairs)
├── Dehaze/        # Dehazing images (hazy/clear pairs)
├── Derain/        # Deraining images (rainy/clean pairs)
├── Deblur/        # Deblurring images (blurred/sharp pairs)
└── Enhance/       # Low-light enhancement images (low-light/normal pairs)
```
## 🚀 Training
### Training Command
```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --epochs 150 \
    --batch_size 8 \
    --num_gpus 4 \
    --de_type denoise_15 denoise_25 denoise_50 dehaze derain deblur enhance \
    --ckpt_dir train_ckpt/DRNet \
    --num_workers 8
```

## 🔍 Inference
### Denoising (5 tasks model)
Test on BSD68 dataset:
```bash
# Denoising σ=15
python inference.py --task "denoise" --ckpt_path ckpt/5tasks/DRNet_5tasks.pth --data_path test/denoise/bsd68/ --sigma 15

# Denoising σ=25
python inference.py --task "denoise" --ckpt_path ckpt/5tasks/DRNet_5tasks.pth --data_path test/denoise/bsd68/ --sigma 25

# Denoising σ=50
python inference.py --task "denoise" --ckpt_path ckpt/5tasks/DRNet_5tasks.pth --data_path test/denoise/bsd68/ --sigma 50
```

### Deraining (5 tasks model)
Test on Rain100L dataset:
```bash
python inference.py --task "derain" --ckpt_path ckpt/5tasks/DRNet_5tasks.pth --data_path test/derain/Rain100L/
```

### Dehazing (3 tasks model)
Test on SOTS dataset:
```bash
python inference.py --task "dehaze" --ckpt_path ckpt/3tasks/DRNet_3tasks.pth --data_path test/dehaze/SOTS/
```

## 📝 Citation
If you find DRNet useful for your research, please cite:
```bibtex
@article{li2026drnet,
  title={DRNet: All-in-One Image Restoration via Prior-Guided Dynamic Reparameterization},
  author={Li, Ao and Liu, Xiaoning and Li, Sheng and Du, Yapeng and Long, Zhen and Luo, Lei and Zhang, Le and Zhu, Ce},
  journal={arXiv preprint arXiv:2605.08627},
  year={2026}
}
```

## 🙏 Acknowledgement
This codebase is built upon [PromptIR](https://github.com/va1shn9v/PromptIR). We thank the authors for their awesome work.
