# Chest X-Ray Pneumonia – ResNet18 (FastAI)

Prototipo de IA para diagnóstico asistido a partir de radiografías de tórax (**NORMAL** vs **PNEUMONIA**).
Implementado con **FastAI**/**PyTorch** usando **transfer learning** (ResNet-18).

## � Resumen rápido
- Modelo: ResNet-18 (transfer learning, FastAI)
- Tareas: Clasificación binaria (NORMAL / PNEUMONIA)
- Artefactos: `project/outputs/export.pkl` (modelo exportado), figuras de evaluación en `project/outputs/`

## �📊 Resultados (ejemplo)
- **Validación**: Accuracy **0.94**, F1 (macro) **0.94**, AUC **1.000**
- **Prueba**: Accuracy **0.94**, F1 (macro) **0.93**, AUC **0.986**
- Figuras: `project/outputs/cm_val.png`, `project/outputs/roc_val.png`, `project/outputs/roc_test.png`

> Dataset: *Chest X-Ray Images (Pneumonia)* (Kaggle – P. Mooney). No se incluyen las imágenes por licencia / tamaño.

---

## 🗂 Estructura del proyecto
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

## ⚙️ Setup (resumen)
Para detalles completos, ver `SETUP.md`. Aquí está el flujo rápido:

1. Crear y activar venv (ejemplo):
```bash
python3 -m venv ia_medica
source ia_medica/bin/activate
```
2. Actualizar pip y herramientas:
```bash
python -m pip install --upgrade pip setuptools wheel
```
3. Instalar PyTorch (ajusta `cu124` según tu CUDA) y dependencias del proyecto:
```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install fastai timm scikit-learn matplotlib jupyterlab
```
4. Verificar CUDA (dentro del venv):
```bash
python - << 'PY'
import torch
print("torch:", torch.__version__, "build CUDA:", torch.version.cuda)
print("cuda available?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

---

## 📦 Dataset
Descargar desde Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
Colocar la carpeta así (no subirla al repo por licencia/size):

```
Datasets/chest_xray/
├── train/{NORMAL,PNEUMONIA}
├── val/{NORMAL,PNEUMONIA}
└── test/{NORMAL,PNEUMONIA}
```

---

## 🚀 Entrenamiento (Quick start)
Desde la raíz del repo, con el venv activado:

```bash
cd project
python train_local.py \
  --data_dir ../Datasets/chest_xray \
  --epochs 8 \
  --bs 32 \
  --num_workers 4
```

Salida esperada: métricas de entrenamiento/validación y artefactos guardados en `project/outputs/`.

Parámetros comunes:
- `--data_dir`: ruta al dataset
- `--epochs`: número de épocas (ej. 8)
- `--bs`: batch size (ej. 32)
- `--num_workers`: workers de data loading (ej. 4)

---

## 🔍 Inferencia (usar modelo entrenado)

1) Usar `export.pkl` generado por FastAI (guardado en `project/outputs/`).

2) Ejemplo de inferencia (archivo `project/infer.py`):

```bash
# Usando export.pkl (ruta por defecto en outputs)
python project/infer.py project/outputs/export.pkl <ruta_a_una_imagen>

# Ejemplo con una imagen del set de prueba:
python project/infer.py ../Datasets/chest_xray/test/PNEUMONIA/person14_virus_44.jpeg
```

Nota: `infer.py` espera la ruta al modelo exportado o usa la ruta por defecto; revisa el script si necesitas pasar parámetros adicionales.

---

## � Outputs generados
- `project/outputs/export.pkl` — modelo exportado (FastAI).
- `project/outputs/cm_val.png`, `roc_val.png`, `roc_test.png` — figuras de evaluación (matriz de confusión, curvas ROC).

---

## 🆘 Troubleshooting (rápido)
- `venv: command not found` → `sudo apt install python3.12-venv`
- `ModuleNotFoundError: No module named 'torch'` → instalar PyTorch dentro del venv
- `cuda available?: False` → revisar `nvidia-smi` y drivers NVIDIA; reinstalar PyTorch con la versión CUDA correcta
- `PermissionError` → no ejecutar como `sudo` dentro del venv

---

## �🧠 Notas finales
- `export.pkl` fue generado con `fastai` (usa `pickle`). No cargar artefactos no confiables.
- Para despliegues seguros, considerar `Learner.save()` o TorchScript.
- Este prototipo es **educativo/investigativo** y **no** está validado para uso clínico.

---

## 👨‍💻 Créditos
- Autor: **Mario Cordero**
- Basado en FastAI + PyTorch