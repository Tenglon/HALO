# HALO: Hyperbolic Adaptation via Lift Overlay

Official code release for:

**HALO: Hyperbolic Adaptation via Lift Overlay for Hierarchy-Aware Cross-Modal Retrieval**  
Teng Long and Andrew Yates, SIGIR 2026 poster.

HALO adapts CLIP-style vision-language models with lightweight LoRA modules and a hyperbolic/Lorentz retrieval objective. This repository contains the HALO training and retrieval evaluation code used for the paper.

## Repository layout

```text
vlm/
  cli/              # training and evaluation entry points
  configs/          # HALO train/eval Hydra configs
  data/             # dataset loading and transforms
  eval/             # COCO/Flickr retrieval evaluation
  geom/             # Euclidean and Lorentz geometry utilities
  losses/           # contrastive retrieval losses
  models/           # CLIP wrapper and LoRA integration
  train/            # training loop and checkpointing
scripts/
  halo_train.sh     # clean HALO training launcher
  halo_eval.sh      # clean HALO checkpoint evaluation launcher
```

The repository may also contain local experiment outputs or third-party baseline folders from development. They are not required for the HALO code path documented here.

## Setup

```bash
conda create -n halo python=3.12 -y
conda activate halo
pip install -r requirements.txt
pip install -e .
```

For multi-GPU training, configure `accelerate` as usual or pass launcher options through the provided scripts.

## Data

Training uses the LLaVA-filtered CC3M-595K dataset from Hugging Face:

```text
pingzhili/llava-filtered-cc3m-595k
```

Evaluation uses:

```text
jxie/coco_captions              # COCO retrieval validation split
lmms-lab/flickr30k              # Flickr30k retrieval test split
```

The evaluation launcher sets `PYTHONNOUSERSITE=1` by default to avoid user-site `datasets` packages shadowing the active environment.

## HALO training configs

Primary HALO configs:

```text
vlm/configs/clip_vit_b32_cc3m_lorentz_lora_negdist.yaml
vlm/configs/clip_vit_b16_cc3m_lorentz_lora_negdist.yaml
vlm/configs/clip_vit_l14_cc3m_lorentz_lora_negdist.yaml
```

Entailment-loss variants:

```text
vlm/configs/clip_vit_b32_cc3m_lorentz_lora_negdist_entail.yaml
vlm/configs/clip_vit_b16_cc3m_lorentz_lora_negdist_entail.yaml
vlm/configs/clip_vit_l14_cc3m_lorentz_lora_negdist_entail.yaml
```

## Train HALO

Example: train HALO with CLIP ViT-B/32 on 8 GPUs.

```bash
scripts/halo_train.sh \
  clip_vit_b32_cc3m_lorentz_lora_negdist \
  outputs/halo_b32
```

Useful overrides:

```bash
NUM_PROCESSES=8 \
PER_GPU_BATCH=256 \
TARGET_GLOBAL_BATCH=2048 \
MAX_STEPS=5000 \
LR=2e-6 \
scripts/halo_train.sh clip_vit_b16_cc3m_lorentz_lora_negdist outputs/halo_b16
```

For ViT-L/14 on 46GB GPUs, use a smaller per-GPU batch:

```bash
NUM_PROCESSES=8 \
PER_GPU_BATCH=128 \
TARGET_GLOBAL_BATCH=2048 \
scripts/halo_train.sh clip_vit_l14_cc3m_lorentz_lora_negdist outputs/halo_l14
```

The launcher writes checkpoints to the output directory, for example:

```text
outputs/halo_b32/checkpoint_step_5000.pt
```

## Evaluate a HALO checkpoint

Evaluate one checkpoint on COCO 5k and Flickr30k 5k:

```bash
scripts/halo_eval.sh \
  vlm/configs/clip_vit_b32_cc3m_lorentz_lora_negdist.yaml \
  outputs/halo_b32/checkpoint_step_5000.pt \
  outputs/eval/halo_b32_step5000
```

Evaluate one dataset only:

```bash
DATASET=coco scripts/halo_eval.sh \
  vlm/configs/clip_vit_b32_cc3m_lorentz_lora_negdist.yaml \
  outputs/halo_b32/checkpoint_step_5000.pt \
  outputs/eval/halo_b32_coco

DATASET=flickr scripts/halo_eval.sh \
  vlm/configs/clip_vit_b32_cc3m_lorentz_lora_negdist.yaml \
  outputs/halo_b32/checkpoint_step_5000.pt \
  outputs/eval/halo_b32_flickr
```

The evaluator prints retrieval metrics in this format:

```python
{'text_to_image': {'r1': ..., 'r5': ..., 'r10': ...}, 'image_to_text': {'r1': ..., 'r5': ..., 'r10': ...}}
```

## Reproducing paper-style retrieval runs

A typical HALO run is:

```bash
NUM_PROCESSES=8 PER_GPU_BATCH=256 TARGET_GLOBAL_BATCH=2048 MAX_STEPS=5000 \
  scripts/halo_train.sh clip_vit_b32_cc3m_lorentz_lora_negdist outputs/halo_b32_5k

scripts/halo_eval.sh \
  vlm/configs/clip_vit_b32_cc3m_lorentz_lora_negdist.yaml \
  outputs/halo_b32_5k/checkpoint_step_5000.pt \
  outputs/eval/halo_b32_5k
```

Use the B/16 or L/14 configs above to reproduce other HALO backbone sizes.

## Citation

```bibtex
@inproceedings{long2026halo,
  title = {HALO: Hyperbolic Adaptation via Lift Overlay for Hierarchy-Aware Cross-Modal Retrieval},
  author = {Long, Teng and Yates, Andrew},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year = {2026}
}
```
