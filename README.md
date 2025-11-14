
# 🩻 Chest X-Ray Pneumonia Detection – ResNet18 (FastAI + PyTorch)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAI](https://img.shields.io/badge/FastAI-2.7+-blueviolet?logo=fastapi&logoColor=white)](https://docs.fast.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🧠 Overview
AI prototype for **medical image diagnosis** based on chest X-rays.  
Trained to classify between **NORMAL** and **PNEUMONIA** using **transfer learning** with a **ResNet18** backbone.

Built with **FastAI** and **PyTorch**, this project demonstrates how deep learning can assist medical professionals by improving diagnostic speed and consistency.

---

## ⚙️ Model Summary
| Component | Description |
|------------|--------------|
| **Framework** | FastAI + PyTorch |
| **Architecture** | ResNet18 pretrained on ImageNet |
| **Learning Strategy** | Transfer Learning + Fine-Tuning |
| **Optimizer** | Adam |
| **Loss Function** | Cross-Entropy |
| **Augmentations** | Rotation, flipping, scaling, brightness, contrast |
| **Normalization** | ImageNet mean/std normalization |

---

## 📊 Results
| Dataset | Accuracy | F1 (macro) | AUC |
|----------|-----------|-------------|-----|
| **Validation** | 0.94 | 0.94 | 1.000 |
| **Test** | 0.94 | 0.93 | 0.986 |

**Generated figures:**  
- `cm_val.png` – Confusion Matrix  
- `roc_val.png` – ROC Curve (Validation)  
- `roc_test.png` – ROC Curve (Test)

---

## 🗂️ Project Structure
```

project/
├── train_local.py          # training + metrics + export.pkl
├── infer.py                # inference with trained model
└── outputs/
├── export.pkl
├── cm_val.png
├── roc_val.png
└── roc_test.png

```

---

## 🧩 Dataset
Dataset used: [**Chest X-Ray Images (Pneumonia)** – Kaggle (Paul Mooney)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Place files as:
```

Datasets/chest_xray/
├── train/{NORMAL,PNEUMONIA}
├── val/{NORMAL,PNEUMONIA}
└── test/{NORMAL,PNEUMONIA}

````

> ⚠️ Images are **not included** in this repository due to license and size limitations.

---

## 🚀 Setup & Training

### 1️⃣ Create and activate a virtual environment
```bash
python3 -m venv ia_medica
source ia_medica/bin/activate
````

### 2️⃣ Install dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install fastai timm scikit-learn matplotlib jupyterlab
```

### 3️⃣ Verify GPU / CUDA availability

```bash
python - << 'PY'
import torch
print("torch:", torch.__version__, "build CUDA:", torch.version.cuda)
print("cuda available?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

### 4️⃣ Train the model

```bash
cd project
python train_local.py \
  --data_dir ../Datasets/chest_xray \
  --epochs 8 \
  --bs 32 \
  --num_workers 4
```

---

## 🔍 Inference Example

Use your trained `export.pkl` model for predictions:

```bash
python project/infer.py project/outputs/export.pkl ../Datasets/chest_xray/test/PNEUMONIA/person14_virus_44.jpeg
```

Example output:

```
Prediction: PNEUMONIA | P(NORMAL)=0.006 | P(PNEUMONIA)=0.994
```

---

## 🧠 Outputs

| File           | Description                     |
| -------------- | ------------------------------- |
| `export.pkl`   | Serialized trained FastAI model |
| `cm_val.png`   | Confusion Matrix (Validation)   |
| `roc_val.png`  | ROC Curve (Validation)          |
| `roc_test.png` | ROC Curve (Test)                |

---

## 🧰 Troubleshooting

| Issue                                          | Fix                                                    |
| ---------------------------------------------- | ------------------------------------------------------ |
| `venv: command not found`                      | `sudo apt install python3.12-venv`                     |
| `ModuleNotFoundError: No module named 'torch'` | Install PyTorch inside your venv                       |
| `cuda available?: False`                       | Check `nvidia-smi` and reinstall matching CUDA version |
| `PermissionError`                              | Avoid running inside venv with `sudo`                  |

---

## ⚠️ Notes

* `export.pkl` uses Python’s **pickle** serialization — only load trusted files.
* For deployment, consider `Learner.save()` or **TorchScript** for safer model export.
* This is a **research and educational** prototype, **not intended for clinical use**.

---

## 👨‍💻 Author

**Mario Cordero**
School of Computer Science and Informatics
University of Costa Rica (UCR)

📍 San José, Costa Rica
🔗 [GitHub Repository](https://github.com/MarioCordero/chest-xray-diagnosis-ai)

---

⭐ *If you find this project useful, please give it a star!*
