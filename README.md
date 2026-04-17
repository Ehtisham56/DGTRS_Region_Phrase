# Supervised Deep Learning in Multi-Scale Region-Phrase Alignment of Remote Sensing Vision Language Models

A deep learning project focused on improving fine-grained alignment between satellite imagery and textual descriptions using multi-scale region-phrase supervision in Vision-Language Models (VLMs). This project is implemented in **PyTorch** and leverages a dual-encoder architecture for remote sensing AI.

---

## 🚀 Features

- Recreation of the DGTRS-CLIP training pipeline
- Region-phrase alignment mechanism for improved fine-grained alignment
- Implementation of local alignment loss
- Enhanced semantic understanding of small satellite objects
- Robustness and generalization evaluation across different datasets
- Supports training under limited supervision, limited datasets, and constrained GPU resources

---

## 🛠 Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- DGTRS-CLIP (Recreation)
- PR-CLIP
- RET-2 Dataset
- Dual Encoder Architecture

---

## 📁 How to Run (Windows)

- Open the project in Visual Studio Code (or your preferred IDE)
- Ensure Python and pip are installed
- Install the required dependencies:
```bash
pip install -r requirements.txt
```
- Run the training pipeline:
```bash
python train.py
```
- Run the evaluation script or notebooks to analyze Recall metrics:
```bash
python eval.py
```

---

## 🧠 Concepts Practiced

This project applies several advanced concepts in deep learning and computer vision:

- Contrastive Learning for Image-Text Retrieval
- Patch-Level Alignment in satellite imagery
- Weak Supervision for fine-grained representation learning
- Vision-Language Alignment
- Evaluating models using Recall@1, Recall@5, and cross-dataset generalization analysis
- Managing practical challenges in VLM training (GPU resources, limited datasets)

---

## 👨‍💻 Author

Ehtisham Abid