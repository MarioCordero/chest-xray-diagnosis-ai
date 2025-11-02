# Chest X-Ray Pneumonia – ResNet18 (FastAI)

Prototipo de IA para diagnóstico asistido a partir de radiografías de tórax (**NORMAL** vs **PNEUMONIA**).  
Implementado con **FastAI**/**PyTorch** usando **transfer learning** (ResNet-18).

## 📊 Resultados
- **Validación**: Accuracy **0.94**, F1 (macro) **0.94**, AUC **1.000**
- **Prueba**: Accuracy **0.94**, F1 (macro) **0.93**, AUC **0.986**
- Figuras: `project/outputs/cm_val.png`, `project/outputs/roc_val.png`, `project/outputs/roc_test.png`

> Dataset: *Chest X-Ray Images (Pneumonia)* (Kaggle – P. Mooney).  
> No se incluyen las imágenes por licencia / tamaño.

---

## 🗂️ Estructura del proyecto
```
project/
├── train_local.py          # entrenamiento + métricas + figuras + export.pkl
├── infer.py                # inferencia con export.pkl
└── outputs/
    ├── export.pkl
    ├── cm_val.png
    ├── roc_val.png
    └── roc_test.png
```

---

## ⚙️ Requisitos
- Python 3.10+  
- PyTorch + CUDA (opcional)  
- FastAI, timm, scikit-learn, matplotlib

> Recomendado crear un venv y usar el índice de PyTorch para instalar con CUDA.

### Instalación (venv)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# PyTorch (CUDA 12.4)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install fastai timm scikit-learn matplotlib
```

---

## 📦 Dataset
Descargar desde Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
Colocar la carpeta así (sin subirla al repo):

```
Datasets/chest_xray/
├── train/{NORMAL,PNEUMONIA}
├── val/{NORMAL,PNEUMONIA}
└── test/{NORMAL,PNEUMONIA}
```

---

## 🚀 Entrenamiento
Desde la raíz del repo:
```bash
python project/train_local.py --data_dir Datasets/chest_xray --epochs 8 --bs 32 --num_workers 4
```

---

## 🔍 Inferencia
```bash
python project/infer.py project/outputs/../ruta/a/una_imagen.jpeg

# Ejemplo con una imagen del set de prueba:
python project/infer.py Datasets/chest_xray/test/PNEUMONIA/person14_virus_44.jpeg
```

---

## 🧠 Notas
- `export.pkl` fue generado con `fastai` (usa `pickle`). No cargar artefactos no confiables.
- Si se desea una versión más segura, exportar con `Learner.save()` o TorchScript.
- Este prototipo es **educativo/investigativo**, **no** validado para uso clínico.

---

## 👨‍💻 Créditos
- Autores: **Mario Cordero**
- Basado en FastAI + PyTorch