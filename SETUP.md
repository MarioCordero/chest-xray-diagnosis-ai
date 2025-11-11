# 🔧 Guía de Configuración del Entorno

Instrucciones detalladas para configurar el entorno de desarrollo del proyecto de diagnóstico de neumonía en radiografías de tórax.

---

## 📥 Paso 1: Descargar el Dataset

Descargar el archivo de Kaggle y descomprimir:  
📍 **Enlace**: [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Colocar el dataset descomprimido en la carpeta `Datasets/` de la raíz del proyecto.

---

## 🖥️ Paso 2: Información del Sistema

Verificar la configuración del sistema local:

```bash
neofetch
```

**Ejemplo de salida esperada:**
```
OS: Linux Mint 22.1 x86_64
Kernel: 6.8.0-87-generic
Shell: bash 5.2.21
CPU: AMD Ryzen 5 5600G with Radeon Graphics (12) @ 4.454GHz
GPU: NVIDIA GeForce RTX 3060 Ti
Memory: 4497MiB / 15873MiB
```

---

## 🐍 Paso 3: Crear el Entorno Virtual

### Instalar python3.12-venv (si no está instalado)
```bash
sudo apt install python3.12-venv
```

### Crear el entorno virtual
```bash
python3 -m venv ia_medica
```

### Activar el entorno
```bash
source ia_medica/bin/activate
```

---

## ✅ Paso 4: Verificar la Instalación del Entorno

Comprobar que el entorno está correctamente activado:

```bash
which python
which pip
python -c "import sys; print(sys.prefix)"
```

**Nota:** La ruta debe apuntar a `.../chest-xray-diagnosis-ai/ia_medica`

---

## 📦 Paso 5: Instalar Dependencias

Dentro del entorno activado, ejecutar:

```bash
python -m pip install --upgrade pip setuptools wheel
```

### Instalar PyTorch con CUDA 12.4
```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Nota:** Si usas CUDA diferente, cambiar `cu124` por tu versión (ej: `cu121`, `cpu`).

### Instalar dependencias del proyecto
```bash
python -m pip install fastai timm scikit-learn matplotlib jupyterlab
```

---

## 🚀 Paso 6: Verificar CUDA

Ejecutar el siguiente script para comprobar que CUDA está correctamente detectado:

```bash
python << 'PY'
import torch
print("torch:", torch.__version__, "build CUDA:", torch.version.cuda)
print("cuda available?:", torch.cuda.is_available())
if torch.cuda.is_available(): 
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

**Salida esperada:**
```
torch: 2.x.x build CUDA: 12.4
cuda available?: True
GPU: NVIDIA GeForce RTX 3060 Ti Lite Hash Rate
```

### ⚠️ Si CUDA no se detecta:
1. Verificar drivers de GPU: `nvidia-smi`
2. Si no funciona, considerar crear un punto de restauración del sistema
3. Reinstalar drivers de NVIDIA (si es necesario)
4. Reinstalar PyTorch con la versión correcta de CUDA

---

## 📁 Paso 7: Verificar Estructura del Proyecto

La estructura debe ser:

```
chest-xray-diagnosis-ai/
├── Datasets/              # Dataset descargado
├── ia_medica/             # Entorno virtual
├── project/               # Scripts de entrenamiento e inferencia
├── notes.txt              # Notas (este archivo)
├── README.md              # Documentación principal
└── SETUP.md               # Este archivo (guía de configuración)
```

Verificar con:
```bash
tree -L 1
```

---

## 🎯 Paso 8: Entrenar el Modelo

Navegar a la carpeta del proyecto y ejecutar el entrenamiento:

```bash
cd project
python train_local.py \
  --data_dir ../Datasets/chest_xray \
  --epochs 8 \
  --bs 32 \
  --num_workers 4
```

**Parámetros:**
- `--data_dir`: Ruta al dataset
- `--epochs`: Número de épocas (8 recomendado)
- `--bs`: Batch size (32 recomendado)
- `--num_workers`: Procesos paralelos (4 recomendado)

El modelo entrenado se guardará en `project/outputs/export.pkl`

---

## 📊 Paso 9: Inferencia

Después del entrenamiento, probar el modelo con una imagen:

```bash
python infer.py <ruta_a_imagen>
```

**Ejemplo:**
```bash
python infer.py ../Datasets/chest_xray/test/PNEUMONIA/person14_virus_44.jpeg
```

---

## 📋 Resumen de Comandos Rápidos

```bash
# Crear y activar entorno
python3 -m venv ia_medica
source ia_medica/bin/activate

# Instalar dependencias
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install fastai timm scikit-learn matplotlib jupyterlab

# Verificar CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Entrenar modelo
cd project
python train_local.py --data_dir ../Datasets/chest_xray --epochs 8 --bs 32 --num_workers 4

# Inferencia
python infer.py ../Datasets/chest_xray/test/PNEUMONIA/person14_virus_44.jpeg
```

---

## 🆘 Troubleshooting

| Problema | Solución |
|----------|----------|
| `venv: command not found` | `sudo apt install python3.12-venv` |
| `ModuleNotFoundError: No module named 'torch'` | Ejecutar instalación de PyTorch |
| `cuda available?: False` | Instalar drivers de NVIDIA o reinstalar PyTorch |
| `PermissionError` | No ejecutar como `sudo` dentro del venv |

---

**¡Listo!** El entorno está configurado y listo para entrenar y hacer inferencia. 🎉