# ANYmal D PPO en MuJoCo

Este repositorio contiene una implementación de **PPO para el robot cuadrúpedo ANYmal D en MuJoCo**. La idea principal del proyecto es tener una base reproducible para entrenar, evaluar y visualizar una política de locomoción usando aprendizaje por refuerzo.

El repo original ya traía el entorno de MuJoCo, el modelo del ANYmal D y scripts de entrenamiento/renderizado. Sobre esa base se agregó soporte para configuración por **YAML**, continuación de entrenamiento desde checkpoints y scripts auxiliares para limpiar resultados viejos y visualizar el mejor modelo.

Este README lo dejé lo más completo posible para que alguien pueda clonar o descomprimir el proyecto en una máquina Linux y saber cómo instalarlo, correrlo, entrenarlo, probar el mejor checkpoint y entender qué archivos importan.

---

## 1. Qué hace este proyecto

El proyecto entrena una política PPO para controlar las 12 articulaciones actuadas del ANYmal D:

- 4 patas.
- 3 articulaciones por pata.
- Acción total: `12` valores.
- Observación del agente: `35` valores.
- Simulador: MuJoCo.
- Red neuronal: PyTorch.
- Tracking de experimentos: Trackio / Hugging Face.
- Configuración principal: `configs/ppo_anymal_d.yaml`.

El script más importante es:

```bash
anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py
```

Ese script permite:

```bash
train   # entrenar PPO
sweep   # hacer búsqueda aleatoria de hiperparámetros
render  # generar videos desde el mejor checkpoint o uno específico
```

---

## 2. Estructura del repositorio

```text
anymal_d_ppo/
├── anybotics_anymal_d/
│   ├── scene.xml
│   ├── anymal_d.xml
│   ├── anymal_d.png
│   └── assets/
│
├── anymal_d/
│   ├── RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN.py
│   ├── RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py
│   └── RL_PPO_ANYMAL_D_VIDEO.py
│
├── configs/
│   └── ppo_anymal_d.yaml
│
├── pretrained_models/
│   └── anymal_d/
│       └── videos/
│
├── clean_checkpoints.py
├── run_best_live.py
├── requirements.txt
└── README.md
```

### Archivos principales

| Archivo | Para qué sirve |
|---|---|
| `anymal_d/RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py` | Script principal. Entrena, hace sweep y genera videos. Es el que más uso. |
| `configs/ppo_anymal_d.yaml` | Configuración del entrenamiento: learning rate, episodios, batch size, gamma, clipping, render, paths, device, etc. |
| `clean_checkpoints.py` | Borra checkpoints viejos y deja solo el mejor, para no consumir tanto espacio. |
| `run_best_live.py` | Abre el mejor checkpoint en una ventana interactiva de MuJoCo. |
| `pretrained_models/anymal_d/` | Aquí se guardan checkpoints `.pt`. |
| `pretrained_models/anymal_d/videos/` | Aquí se guardan videos `.mp4` y gráficas `.png`. |
| `anybotics_anymal_d/scene.xml` | Escena principal de MuJoCo para cargar el robot. |

---

## 3. Requisitos generales

Esto está pensado para Linux. Lo probé con un flujo tipo Ubuntu + Conda, pero debería funcionar en cualquier distro si se instalan las dependencias equivalentes.

### Hardware recomendado

- CPU moderna.
- RAM: mínimo 8 GB, mejor 16 GB o más.
- GPU NVIDIA opcional, pero muy recomendable.
- Driver NVIDIA funcionando si se quiere entrenar con CUDA.

Para revisar si hay GPU NVIDIA:

```bash
nvidia-smi
```

Si ese comando muestra tu GPU y versión del driver, entonces PyTorch debería poder usar CUDA si se instala la versión correcta.

### Software recomendado

- Linux.
- Python 3.10 o superior.
- Conda o Miniconda.
- Git.
- MuJoCo Python bindings.
- PyTorch.

---

## 4. Instalación desde cero en Linux

### 4.1. Instalar paquetes del sistema

En Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y git python3-pip python3-venv build-essential \
    libglfw3 libglfw3-dev libosmesa6 libosmesa6-dev \
    libglew-dev patchelf ffmpeg
```

Si usas otra distro, instala los equivalentes de:

```text
git
build-essential / base-devel
glfw
osmesa
glew
patchelf
ffmpeg
```

`ffmpeg` es útil porque los videos generados por MuJoCo/mediapy normalmente terminan como `.mp4`.

---

## 5. Crear entorno de Python

Yo recomiendo usar Conda para no mezclar dependencias con el sistema.

```bash
conda create -n anymal_ppo python=3.11 -y
conda activate anymal_ppo
```

Actualiza herramientas base:

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## 6. Instalar PyTorch

### Opción A: con GPU NVIDIA / CUDA

Primero revisa que el sistema vea la GPU:

```bash
nvidia-smi
```

Luego instala PyTorch con soporte CUDA. La versión exacta puede cambiar con el tiempo, por eso lo más seguro es revisar el selector oficial de PyTorch. Un ejemplo típico con CUDA 12.6 es:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Si tu máquina usa otra versión de CUDA compatible con PyTorch, cambia el índice según el comando que te dé la página oficial.

### Opción B: solo CPU

Si no tienes GPU NVIDIA, también se puede correr en CPU, pero el entrenamiento será más lento:

```bash
pip install torch torchvision torchaudio
```

---

## 7. Instalar dependencias del repo

Desde la raíz del proyecto:

```bash
cd ~/VISUAL/Pruebaconda/anymal_d_ppo
pip install -r requirements.txt
```

Si prefieres instalar manualmente:

```bash
pip install mujoco numpy mediapy matplotlib trackio huggingface_hub PyOpenGL pyyaml tqdm
```

Para verificar que todo quedó bien:

```bash
python - <<'PY'
import torch
import mujoco
import yaml
print("Torch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("MuJoCo:", mujoco.__version__)
print("YAML OK")
PY
```

Si todo está bien, debería decir `CUDA disponible: True` en una máquina con GPU NVIDIA configurada. En CPU dirá `False`, y eso no necesariamente es error.

---

## 8. Forma recomendada de correr los scripts

Aunque también se pueden ejecutar como archivo directo, recomiendo correrlos como módulo con `PYTHONPATH=$PWD`. Esto evita problemas cuando PyTorch carga checkpoints guardados con clases dentro del paquete `anymal_d`.

Desde la raíz del repo:

```bash
cd ~/VISUAL/Pruebaconda/anymal_d_ppo
conda activate anymal_ppo
```

Usa siempre este estilo:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING <modo> [opciones]
```

Por ejemplo:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING train
```

---

## 9. Configuración YAML

Una de las mejoras importantes de este repo es que los parámetros del entrenamiento se movieron a un archivo YAML:

```bash
configs/ppo_anymal_d.yaml
```

Contenido actual:

```yaml
project:
  name: AIDL-PPO-ANYMAL_D
  run_name: yaml_train

device: auto

training:
  num_episodes: 15000
  lr: 0.0003
  gamma: 0.99
  batch_size: 128
  clip_param: 0.2
  ppo_epoch: 10
  replay_size: 4096
  c1: 0.5
  c2: 0.005
  max_grad_norm: 0.5
  log_interval: 50

checkpointing:
  checkpoint_freq: 0
  save_best_only: true

rendering:
  live: false
  render_on_checkpoint: true
  fall_threshold: 0.35

output:
  save_dir: pretrained_models/anymal_d
  video_dir: pretrained_models/anymal_d/videos
```

### Qué significa cada sección

#### `project`

```yaml
project:
  name: AIDL-PPO-ANYMAL_D
  run_name: yaml_train
```

Define el nombre del proyecto en Trackio y el nombre base del experimento. El `run_name` también se usa para nombrar checkpoints.

#### `device`

```yaml
device: auto
```

Opciones útiles:

```yaml
device: auto
```

Usa GPU si está disponible, si no usa CPU.

```yaml
device: cuda
```

Fuerza CUDA.

```yaml
device: cpu
```

Fuerza CPU.

#### `training`

Aquí viven los hiperparámetros de PPO:

| Parámetro | Significado |
|---|---|
| `num_episodes` | Número máximo de episodios de entrenamiento. |
| `lr` | Learning rate. |
| `gamma` | Discount factor. |
| `batch_size` | Tamaño de batch usado en PPO. |
| `clip_param` | Parámetro de clipping de PPO. |
| `ppo_epoch` | Número de pasadas de actualización por batch. |
| `replay_size` | Tamaño de memoria antes de actualizar. |
| `c1` | Peso de la pérdida del crítico. |
| `c2` | Peso de la entropía. |
| `max_grad_norm` | Clipping de gradiente. |
| `log_interval` | Cada cuántos episodios imprime logs. |

#### `checkpointing`

```yaml
checkpointing:
  checkpoint_freq: 0
  save_best_only: true
```

- `checkpoint_freq: 0` significa que no se guardan checkpoints periódicos.
- El script guarda automáticamente cuando mejora el `running_reward`.
- Si pones `checkpoint_freq: 100`, además de los mejores, guardará checkpoints periódicos cada 100 episodios.

#### `rendering`

```yaml
rendering:
  live: false
  render_on_checkpoint: true
  fall_threshold: 0.35
```

- `live`: abre viewer de MuJoCo durante entrenamiento si está en `true`.
- `render_on_checkpoint`: genera video cada vez que guarda un checkpoint mejor.
- `fall_threshold`: altura mínima antes de considerar que el robot cayó.

#### `output`

```yaml
output:
  save_dir: pretrained_models/anymal_d
  video_dir: pretrained_models/anymal_d/videos
```

Define dónde se guardan checkpoints y videos.

---

## 10. Probar que el YAML está conectado

Para revisar que el script reconoce `--config`:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING -h | grep config
```

Debe aparecer algo parecido a:

```text
[--config CONFIG]
--config CONFIG       Path to YAML configuration file for training parameters.
```

Para hacer una prueba corta sin esperar horas:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING train \
  --config configs/ppo_anymal_d.yaml \
  --episodes 20 \
  --run-name yaml_test
```

Salida esperada al inicio:

```text
Loaded YAML config from: .../configs/ppo_anymal_d.yaml
Configured device: cuda
Configured checkpoint directory: .../pretrained_models/anymal_d
Configured video directory: .../pretrained_models/anymal_d/videos
```

Si aparece eso, entonces la configuración YAML sí se está usando.

---

## 11. Entrenar desde cero

Entrenamiento normal usando YAML:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING train \
  --config configs/ppo_anymal_d.yaml \
  --run-name yaml_full_train
```

Para una prueba más corta:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING train \
  --config configs/ppo_anymal_d.yaml \
  --episodes 100 \
  --run-name yaml_100_ep
```

Con viewer de MuJoCo durante entrenamiento:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING train \
  --config configs/ppo_anymal_d.yaml \
  --run-name yaml_live_train \
  --live
```

Nota: entrenar con `--live` se ve más bonito, pero puede ir más lento. Para entrenar en serio conviene dejarlo sin viewer y solo generar videos.

---

## 12. Continuar entrenamiento desde el mejor checkpoint

El script también permite continuar desde un checkpoint existente. Esto evita empezar de cero.

Para continuar desde el mejor checkpoint detectado automáticamente:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING train \
  --config configs/ppo_anymal_d.yaml \
  --resume-policy best \
  --episodes 3000 \
  --lr 1e-4 \
  --run-name resume_best
```

Para probarlo rápido:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING train \
  --config configs/ppo_anymal_d.yaml \
  --resume-policy best \
  --episodes 100 \
  --lr 1e-4 \
  --run-name test_resume
```

También puedes pasar una ruta específica:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING train \
  --config configs/ppo_anymal_d.yaml \
  --resume-policy pretrained_models/anymal_d/NOMBRE_DEL_CHECKPOINT_policy.pt \
  --episodes 1000 \
  --lr 1e-4 \
  --run-name resume_specific
```

### Cuándo conviene usar `--resume-policy best`

Conviene cuando ya tienes un checkpoint decente y quieres seguir entrenando sin perder lo aprendido. Normalmente uso un learning rate menor, por ejemplo:

```bash
--lr 1e-4
```

o incluso:

```bash
--lr 5e-5
```

Más episodios no siempre significa mejor, pero sí da más oportunidades de mejorar. En PPO puede estancarse o incluso empeorar, por eso el script guarda checkpoints cuando el promedio mejora.

---

## 13. Probar el mejor modelo y generar videos

Para renderizar el mejor checkpoint:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING render --num-videos 3
```

Esto busca automáticamente el `*_policy.pt` con mayor reward en:

```text
pretrained_models/anymal_d/
```

Los videos salen en:

```text
pretrained_models/anymal_d/videos/
```

Para abrir la carpeta:

```bash
xdg-open pretrained_models/anymal_d/videos
```

O con Nautilus:

```bash
nautilus pretrained_models/anymal_d/videos &
```

Para reproducir con VLC:

```bash
vlc pretrained_models/anymal_d/videos/*.mp4
```

Si no tienes VLC:

```bash
sudo apt install -y vlc
```

### Renderizar un checkpoint específico

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING render \
  --policy pretrained_models/anymal_d/NOMBRE_DEL_CHECKPOINT_policy.pt \
  --num-videos 3
```

---

## 14. Ver el mejor checkpoint en ventana interactiva de MuJoCo

Para abrir el mejor modelo en una ventana de MuJoCo:

```bash
PYTHONPATH=$PWD python run_best_live.py
```

Este script:

1. Busca el mejor checkpoint.
2. Carga la política.
3. Abre el viewer de MuJoCo.
4. Reinicia el episodio cada vez que el robot cae o termina.

Si no abre ventana, puede ser un problema de display/OpenGL. Ver la sección de troubleshooting.

---

## 15. Ver resultados del entrenamiento

### Listar checkpoints

```bash
find pretrained_models/anymal_d -name "*_policy.pt" | sort
```

### Contar checkpoints

```bash
find pretrained_models/anymal_d -name "*_policy.pt" | wc -l
```

### Ver cuánto pesa la carpeta de resultados

```bash
du -sh pretrained_models/anymal_d
```

### Ver videos generados

```bash
find pretrained_models/anymal_d/videos -name "*.mp4" | sort
```

### Abrir la carpeta de videos

```bash
xdg-open pretrained_models/anymal_d/videos
```

---

## 16. Limpiar checkpoints viejos

Como el entrenamiento puede generar muchos `.pt` y videos, agregué:

```bash
clean_checkpoints.py
```

Primero corre en modo seguro para ver qué borraría:

```bash
python clean_checkpoints.py
```

Si todo se ve bien, borra todos los checkpoints excepto el mejor:

```bash
python clean_checkpoints.py --yes
```

Si también quieres borrar videos e imágenes:

```bash
python clean_checkpoints.py --yes --delete-videos
```

Después revisa espacio:

```bash
du -sh pretrained_models/anymal_d
```

---

## 17. Trackio y dashboard local

El entrenamiento usa Trackio como reemplazo tipo `wandb`.

Cuando corres el entrenamiento, normalmente aparece algo como:

```text
Trackio project initialized: AIDL-PPO-ANYMAL_D
Trackio metrics logged to: ~/.cache/huggingface/trackio
```

Para abrir el dashboard local:

```bash
trackio show --project "AIDL-PPO-ANYMAL_D"
```

Esto sirve para ver métricas como:

- reward promedio,
- policy loss,
- value loss,
- entropía,
- ratio de PPO,
- action std.

---

## 18. Integración opcional con Hugging Face

El repo puede subir checkpoints a Hugging Face si configuras estas variables:

```bash
export HF_MODEL_REPO="tu_usuario/anymal-d-ppo"
export HF_TOKEN="hf_tu_token"
export HF_PRIVATE=1
```

Si no defines esas variables, no pasa nada: todo funciona localmente. El upload se salta automáticamente.

También se puede usar:

```bash
export TRACKIO_SPACE_ID="tu_usuario/anymal-d-dash"
```

si se quiere conectar Trackio con un Space.

---

## 19. Sweep de hiperparámetros

Para correr un sweep corto:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING sweep \
  --sweep-count 2 \
  --sweep-episodes 50
```

Para uno más largo:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING sweep \
  --sweep-count 30 \
  --sweep-episodes 2000
```

El sweep prueba variaciones de:

- `lr`
- `ppo_epoch`
- `clip_param`
- `c2`
- `replay_size`

---

## 20. Comandos rápidos que más uso

### Activar entorno

```bash
cd ~/VISUAL/Pruebaconda/anymal_d_ppo
conda activate anymal_ppo
```

### Probar instalación

```bash
python - <<'PY'
import torch, mujoco, yaml
print("CUDA:", torch.cuda.is_available())
print("MuJoCo OK")
print("YAML OK")
PY
```

### Probar YAML con 20 episodios

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING train \
  --config configs/ppo_anymal_d.yaml \
  --episodes 20 \
  --run-name yaml_test
```

### Entrenar completo

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING train \
  --config configs/ppo_anymal_d.yaml \
  --run-name yaml_full_train
```

### Continuar desde el mejor

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING train \
  --config configs/ppo_anymal_d.yaml \
  --resume-policy best \
  --episodes 3000 \
  --lr 1e-4 \
  --run-name resume_best
```

### Renderizar mejor checkpoint

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING render --num-videos 3
```

### Abrir videos

```bash
xdg-open pretrained_models/anymal_d/videos
```

### Ver mejor en MuJoCo live

```bash
PYTHONPATH=$PWD python run_best_live.py
```

### Limpiar checkpoints

```bash
python clean_checkpoints.py
python clean_checkpoints.py --yes
```

---

## 21. Evidencia de que Feature 4 funciona

La parte que se necesitaba para la rúbrica era:

> Move training parameters to YAML.

En este repo ya se cumple porque:

1. Existe un archivo YAML:

```bash
configs/ppo_anymal_d.yaml
```

2. El script acepta `--config`:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING -h | grep config
```

3. El entrenamiento carga la configuración:

```text
Loaded YAML config from: .../configs/ppo_anymal_d.yaml
Configured device: cuda
Configured checkpoint directory: .../pretrained_models/anymal_d
Configured video directory: .../pretrained_models/anymal_d/videos
```

4. El YAML incluye, como mínimo:

- learning rate: `lr`
- episodios: `num_episodes`
- batch size: `batch_size`
- discount factor: `gamma`
- PPO clipping: `clip_param`
- frecuencia de checkpoint: `checkpoint_freq`
- render settings: `live`, `render_on_checkpoint`, `fall_threshold`
- output directories: `save_dir`, `video_dir`
- device configuration: `device`

Con eso se cubre directamente la parte de la rúbrica de configuración YAML.

---

## 22. Prioridad de configuración

El script combina parámetros en este orden:

1. Valores base dentro del script.
2. Valores del YAML (`--config`).
3. Overrides de línea de comandos, por ejemplo `--episodes`, `--lr`, `--run-name`.

Entonces si el YAML dice:

```yaml
training:
  num_episodes: 15000
```

pero corres:

```bash
--episodes 20
```

se usarán 20 episodios para esa corrida.

---

## 23. Troubleshooting

### Error: `ModuleNotFoundError: No module named 'anymal_d'`

Usa `PYTHONPATH=$PWD` y corre el script como módulo:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING render --num-videos 3
```

Esto es especialmente importante al cargar checkpoints.

---

### Error: `Can't get attribute 'Agent'`

Pasa cuando PyTorch intenta cargar un checkpoint guardado con una ruta de clase distinta. La solución práctica es correr desde la raíz del repo como módulo:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING render --num-videos 3
```

Para `run_best_live.py`, correr así:

```bash
PYTHONPATH=$PWD python run_best_live.py
```

---

### Error al abrir MuJoCo viewer

Prueba primero sin viewer:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING train \
  --config configs/ppo_anymal_d.yaml \
  --episodes 20
```

Si necesitas render headless:

```bash
export MUJOCO_GL=osmesa
```

O para viewer con ventana:

```bash
export MUJOCO_GL=glfw
```

También revisa que estén instaladas dependencias de OpenGL:

```bash
sudo apt install -y libglfw3 libglfw3-dev libosmesa6 libosmesa6-dev libglew-dev patchelf
```

---

### `CUDA disponible: False`

Revisa:

```bash
nvidia-smi
```

Si `nvidia-smi` no funciona, el problema está en el driver NVIDIA, no en el repo.

Si `nvidia-smi` sí funciona, probablemente instalaste PyTorch CPU. Reinstala PyTorch con el comando CUDA correspondiente desde el selector oficial de PyTorch.

---

### `configs/ppo_anymal_d.yaml: Permiso denegado`

Eso pasa si intentas ejecutar el YAML como si fuera un script:

```bash
configs/ppo_anymal_d.yaml
```

No se ejecuta así. Para verlo:

```bash
cat configs/ppo_anymal_d.yaml
```

Para editarlo:

```bash
nano configs/ppo_anymal_d.yaml
```

Para usarlo en entrenamiento:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING train \
  --config configs/ppo_anymal_d.yaml
```

---

### No se generan checkpoints

El script no guarda en cada episodio por default. Guarda cuando el `running_reward` supera el mejor valor guardado.

Para forzar checkpoints periódicos, cambia en el YAML:

```yaml
checkpointing:
  checkpoint_freq: 100
```

Así guardará cada 100 episodios además de los checkpoints por mejora.

---

### Se generan demasiados videos/checkpoints

Puedes apagar videos en checkpoints:

```yaml
rendering:
  render_on_checkpoint: false
```

También puedes limpiar resultados viejos:

```bash
python clean_checkpoints.py --yes --delete-videos
```

---

## 24. Notas sobre los scripts originales

El repo trae tres scripts de entrenamiento/renderizado:

### `RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN.py`

Versión más vieja. Usa una política distinta basada en `MultivariateNormal`. Sirve como referencia, pero no es la que recomiendo para trabajar ahora.

### `RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING.py`

Versión más completa. Tiene:

- entrenamiento,
- sweep,
- render,
- videos,
- Trackio,
- Hugging Face opcional,
- YAML config,
- resume desde checkpoint.

Esta es la versión recomendada.

### `RL_PPO_ANYMAL_D_VIDEO.py`

Script para videos de la versión anterior. Para el flujo actual recomiendo usar mejor:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING render --num-videos 3
```

---

## 25. Qué archivos no conviene subir a GitHub

Los checkpoints y videos pueden pesar bastante. Normalmente no conviene subir esto:

```text
pretrained_models/anymal_d/*.pt
pretrained_models/anymal_d/videos/*.mp4
pretrained_models/anymal_d/videos/*.png
__pycache__/
*.pyc
```

Si se quieren compartir modelos, mejor usar Hugging Face Hub o subir solo el mejor checkpoint.

---

## 26. Referencias útiles

- PyTorch install selector: https://pytorch.org/get-started/locally/
- MuJoCo Python docs: https://mujoco.readthedocs.io/en/stable/python.html
- MuJoCo project: https://mujoco.org/
- Hugging Face Hub: https://huggingface.co/docs/hub/index

---

## 27. Resumen rápido

Para alguien que solo quiere correrlo rápido:

```bash
cd ~/VISUAL/Pruebaconda/anymal_d_ppo
conda activate anymal_ppo
pip install -r requirements.txt
```

Prueba corta:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING train \
  --config configs/ppo_anymal_d.yaml \
  --episodes 20 \
  --run-name yaml_test
```

Renderizar el mejor:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING render --num-videos 3
```

Abrir videos:

```bash
xdg-open pretrained_models/anymal_d/videos
```

Continuar desde el mejor:

```bash
PYTHONPATH=$PWD python -m anymal_d.RL_PPO_ANYMAL_D_SWEEP_OR_TRAIN_RENDERING train \
  --config configs/ppo_anymal_d.yaml \
  --resume-policy best \
  --episodes 3000 \
  --lr 1e-4 \
  --run-name resume_best
```

Con eso ya se puede instalar, entrenar, evaluar, visualizar y documentar la Feature 4 de YAML.
