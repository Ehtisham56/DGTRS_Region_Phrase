# DGTRS Region-Phrase

DGTRS Region-Phrase extends DGTRS-CLIP with local region-phrase alignment for remote sensing image-text retrieval.

In addition to global image-caption contrastive learning, this project aligns image patch embeddings with phrase-level text embeddings extracted from captions. The result is a multi-scale training objective that improves fine-grained cross-modal matching.

## Highlights

- Dual-level alignment:
  - Global: image embedding <-> full caption embedding
  - Local: patch embeddings <-> phrase embeddings
- LongCLIP-compatible training pipeline
- VRAM-safe launcher with automatic retry over batch size and accumulation settings
- Retrieval evaluation with Recall@1/5/10 for both I2T and T2I

## Repository Structure

```text
.
|- config.py
|- dataset_loader.py
|- train.py
|- train_vram_safe.py
|- eval.py
|- run_demo.py
|- loss/
|  |- contrastive_loss.py
|- model/
|  |- dgtrs_clip.py
|  |- dgtrs_longclip.py
|  |- longclip.py
|- dataset/
|  |- train/
|  |  |- ret2.csv
|  |- test/
|     |- rsitmd_test.csv
|     |- rsitmd_val.csv
|     |- rsicd_test.csv
|     |- rsicd_val.csv
|- img/
```

## Requirements

Install dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

Main dependencies:

- torch >= 2.1
- torchvision >= 0.16
- transformers >= 4.37
- pillow, numpy, ftfy, regex, tqdm

## Dataset Setup

This code expects caption CSV files with the following columns:

- filename
- title

Example row:

```csv
filename,title
baseballfield_452.tif,There is a baseball field beside the green amusement park.
```

Image path behavior:

- Primary image root is images (config default).
- Fallback root includes img by default.

If your files are in img, training will still work because img is searched as a fallback root.

## Training

### Standard training

```bash
python train.py \
  --train_csv dataset/train/ret2.csv \
  --model_family longclip_approx \
  --tokenization_mode longclip \
  --normalization_mode clip \
  --batch_size 4 \
  --accumulation_steps 4 \
  --epochs 3 \
  --run_name dgtrs_longclip_region_phrase
```

### VRAM-safe training (recommended on limited GPUs)

```bash
python train_vram_safe.py \
  --start_batch_size 4 \
  --start_accumulation_steps 8 \
  --min_batch_size 1 \
  --max_accumulation_steps 32 \
  --image_sizes 224,192,160 \
  --model_family longclip_approx \
  --tokenization_mode longclip \
  --normalization_mode clip \
  --optimizer_name sgd \
  --lr 5e-4 \
  --momentum 0.9 \
  --dampening 0.1 \
  --use_dgcl true \
  --long_context_length 248 \
  --short_truncate_length 20 \
  --epochs 20 \
  --num_workers 2 \
  --device cuda \
  --run_name dgtrs_longclip_region_phrase
```

Windows PowerShell line continuation uses backtick (`) instead of backslash.

Outputs are saved under checkpoints/<run_name>_<timestamp>, including:

- last_model.pt
- best_model.pt
- config.json

## Evaluation

Run retrieval evaluation on a checkpoint:

```bash
python eval.py \
  --checkpoint checkpoints/<run_dir>/best_model.pt \
  --train_csv dataset/train/ret2.csv \
  --val_csv dataset/test/rsitmd_val.csv \
  --model_family longclip_approx \
  --tokenization_mode longclip \
  --normalization_mode clip \
  --batch_size 4 \
  --device cuda
```

Reported metrics include:

- Recall@1, Recall@5, Recall@10
- Image-to-Text R@K and Text-to-Image R@K

## Demo Inference

Use the quick demo script for image-text matching:

```bash
python run_demo.py
```

Before running:

- Place a trained checkpoint at checkpoints/best_model.pt, or update the path in run_demo.py.
- Place your demo image at img/demo.jpg, or update the image path in run_demo.py.

## Notes

- model_family longclip_approx requires tokenization_mode longclip and normalization_mode clip.
- If val_csv is not provided during training, train.py creates a validation split from train_csv using val_split.
- For reproducibility, set seed and keep dataset paths fixed across runs.

## Acknowledgments

This implementation is inspired by DGTRS-CLIP, LongCLIP, and remote sensing retrieval benchmarks such as RSITMD.