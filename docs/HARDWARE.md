# Entorno de ejecución real

Los resultados del TFM se obtuvieron en la siguiente estación de trabajo y
el código del repositorio está ajustado a esas capacidades. Todo lo demás
(notebooks, scripts, tests) puede correr en máquinas más modestas, pero las
configuraciones por defecto asumen este perfil.

## Workstation

| Componente | Especificación |
|---|---|
| **CPU** | AMD Ryzen 9 5950X (16 núcleos / 32 hilos, Zen 3) |
| **GPU** | NVIDIA GeForce RTX 5090 — arquitectura **Blackwell** (SM 12.0) |
| **VRAM** | 32 GB GDDR7 (disponibles ≈ 33.6 GB reportados por el driver) |
| **RAM** | 64 GB+ recomendada |
| **Almacenamiento** | NVMe (dataset MILK10k ≈ 45 GB con imágenes clínicas + dermatoscópicas) |

## Stack de software validado

| Pieza | Versión en producción |
|---|---|
| Ubuntu / kernel | Linux 6.17 |
| NVIDIA driver | ≥ 570 (obligatorio para Blackwell) |
| CUDA | 12.8+ (validado con build CUDA 13.0) |
| Python | 3.11 |
| PyTorch | **2.9.0+cu130** (mínimo recomendado 2.7 para soporte Blackwell estable) |
| timm | ≥ 0.9.12 |
| torch-geometric | ≥ 2.4 |
| albumentations | ≥ 1.3 (recomendado 2.x con API `num_holes_range`) |

> RTX 5090 (SM 12.0) sólo es reconocida por binarios compilados para
> sm_100 / sm_120. Evita ruedas de PyTorch anteriores a 2.7 o instalaciones
> CUDA < 12.8.

## Precisión y AMP

La Blackwell añade **BF16 nativo a tasa pico**, y recomendamos usarlo sobre
FP16 para el entrenamiento:

- `amp_dtype: bf16` en los YAML de ambas versiones.
- No se usa `torch.cuda.amp.GradScaler` cuando la precisión mixta es BF16
  (el rango dinámico de BF16 evita que los gradientes se desborden).
- El API moderno `torch.amp.autocast("cuda", dtype=torch.bfloat16)` es el
  que usa el Trainer.

FP8 (E4M3/E5M2) queda fuera del alcance del TFM; está disponible en la
tarjeta pero sin beneficios claros en ViT-L/16 a 224 × 224.

## Ajustes por versión

### v1.9 (ganadora)

| Parámetro | Valor por defecto | Nota sobre la RTX 5090 |
|---|---|---|
| batch_size | 128 | ~22 GB con BF16 + gradient checkpointing off |
| image_size | 224 | El ViT-L/16 no se beneficia de 336² |
| num_workers | 8 | 16 procesos saturan la NVMe pero poco la CPU |
| amp / amp_dtype | true / bf16 | BF16 nativo |
| torch.compile | deshabilitado por defecto | Actívalo si buscas +15 % throughput |

### v3.12.x (experimental)

- batch_size = 96: las 11 proxys de ProxyNCA++ + 11 nodos GAT elevan la
  memoria activa ~ 30 % frente a v1.9. 96 deja margen en los picos.
- Unfreeze gradual hasta 6 bloques: con 6 descongelados la memoria pico
  ronda 28 GB (sin `gradient_checkpointing`).

## Pesos PanDerm

En la misma carpeta ``ISIC_25-26/checkpoints/`` hay dos checkpoints MAE:

| Fichero | Arquitectura | Tamaño | Params |
|---|---|---|---|
| **`panderm_ll_data6_checkpoint-499.pth`** | **ViT-L/16** | 1.4 GB | 303.3 M |
| `panderm_bb_data6_checkpoint-499.pth` | ViT-B/16 | 343 MB | 86 M |

El ViT-L es el que usa la memoria TFM. Ambos son checkpoints **MAE pretrained**
con convención **BEiT** (q/v bias separados, LayerScale vía `gamma_1`/`gamma_2`,
claves con prefijo `encoder.*`). El loader `PanDermViT._remap_panderm_sd`:

- descarta claves MAE-only (`mask_token`, `rd_pos_embed`, `decoder.*`, `regresser.*`);
- elimina el prefijo `encoder.`;
- remapea `blocks.N.gamma_1|2 → blocks.N.ls1|2.gamma` (LayerScale en timm);
- concatena `attn.q_bias` + `attn.v_bias` en `attn.qkv.bias` con k-bias = 0;
- filtra por shape para soportar ViT-B → ViT-L parcial si fuese necesario.

El YAML por defecto apunta al ViT-L y fija `layer_scale_init: 1.0` (necesario
para que timm cree los módulos LayerScale que recibirán las `gamma`).
`configs/panderm_vitb.yaml` (sólo en v1.9) carga el ViT-B como alternativa.

Diagnóstico al cargar (captura de pytest):

```
[PanDermViT] loaded=342/342 skipped_shape=0 missing=0 unexpected=0   ← ViT-L
[PanDermViT] loaded=174/174 skipped_shape=0 missing=0 unexpected=0   ← ViT-B
```

## Reproducibilidad

- `train.torch_compile: false` por defecto (los pases compilados pueden
  introducir diferencias numéricas con cuDNN non-determinism).
- `set_seed(2025)` al inicio de cada script (v. `src/utils.py`).
- `torch.backends.cudnn.benchmark = True` para acelerar a cambio de perder
  determinismo estricto — el TFM usa esto.

## Tiempo real observado (referencia de la memoria TFM)

| Operación | Hardware | Tiempo |
|---|---|---|
| 1 época · batch 128 · v1.9 | RTX 5090 + 5950X | ~ 9 min |
| 5-fold CV completo v1.9 | RTX 5090 + 5950X | ~ 6-8 h |
| 5-fold CV completo v3.12.x | RTX 5090 + 5950X | ~ 9-10 h |
| Inferencia test + TTA D4 (8×) | RTX 5090 | ~ 12 min |
