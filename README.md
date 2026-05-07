# ISIC_25-26: Clasificador Multimodal Híbrido ViT-GAT

> ## Aviso de acceso · Access notice
>
> 🇪🇸 Este repositorio contiene únicamente la **documentación pública** del proyecto (memoria PDF, diagramas, overviews y resúmenes técnicos). El **código fuente** permanece en un repositorio privado mientras dure el desafío **ISIC MILK10k 2025** y se abrirá al finalizar el reto. Si necesitas acceso autorizado al código antes de la apertura, escribe a **mmunozpl@uoc.edu**.
>
> 🇬🇧 This repository hosts only the project's **public documentation** (memoir PDF, architecture diagrams, technical overviews and summaries). The **source code** is kept in a private repository for the duration of the **ISIC MILK10k 2025** challenge and will be released publicly once the challenge concludes. If you require authorized access to the source code before then, please contact **mmunozpl@uoc.edu**.

**Proyecto:** `ISIC_25-26` — Trabajo Fin de Máster 2025-26 con participación real en el desafío ISIC MILK10k 2025. Diseño e implementación originales.

**Autor:** Manuel Muñoz Plá

**Institución:** UOC (Universitat Oberta de Catalunya)

**Fecha:** Diciembre 2025

**Challenge:** [ISIC MILK10k 2025](https://challenge.isic-archive.com/leaderboards/milk10k/) — Posición **7 / 115** equipos a fecha de mayo de 2026 (en el leaderboard como `ManuelPla`).


## Documentación y recursos

- **Memoria oficial (PDF)**: [Clasificador Multimodal Híbrido ViT-GAT — Memoria Dic 2025](<docs/Clasificador Multimodal Híbrido ViT-GAT Memoria Dic 2025 - Manuel Muñoz Plá.pdf>)
- **Arquitectura v1.9** (Figura 3.2 del TFM): [`architecture_v1_9.png`](docs/assets/architecture_v1_9.png)
- **Arquitectura v3.12.x** (Figura 9.1 del TFM): [`architecture_v3_12.png`](docs/assets/architecture_v3_12.png)
- **Overview v1.9 (autocontenido)**: [`PanDerm.GAT.V.1.9.PEC3.md`](docs/PanDerm.GAT.V.1.9.PEC3.md)
- **Overview v3.12.x (autocontenido)**: [`Vit-GAT_v3_12_1_M4_FINAL.md`](docs/Vit-GAT_v3_12_1_M4_FINAL.md)
- **Resumen ejecutivo**: [`TFM_Resumen_Ejecutivo.md`](docs/TFM_Resumen_Ejecutivo.md)
- **Hardware, tiempos y ajustes**: [`HARDWARE.md`](docs/HARDWARE.md)
- **Comparativa v1.9 vs v3.12.x**: [`COMPARISON.md`](docs/COMPARISON.md)
- **Resultados detallados**: [`RESULTS.md`](docs/RESULTS.md)

Este repositorio contiene las dos versiones del clasificador multimodal híbrido
ViT-GAT descritas en la memoria de TFM "Clasificador Multimodal Híbrido ViT-GAT
Memoria TFM Dic 2025 – Manuel Muñoz Plá". Cada versión es un paquete Python
independiente, modularizado y ejecutable desde terminal, y además incluye los
notebooks Jupyter (`.ipynb`) equivalentes.

## Estructura general

```
ISIC_25-26/
├── README.md
├── LICENSE · .gitignore · requirements.txt · requirements-dev.txt
│
├── v1.9_monet_gat/         ← Versión original (F1 LB = 0.478)
│   ├── src/                ← Módulos Python reutilizables
│   ├── scripts/            ← Entradas CLI (train, evaluate, infer)
│   ├── notebooks/          ← Jupyter notebooks equivalentes
│   ├── configs/            ← Configuración YAML del modelo
│   ├── data/               ← Metadatos / CSV de entrenamiento
│   ├── docs/               ← Documentación específica de la versión
│   └── outputs/            ← Checkpoints, logs, predicciones
│
├── v3.12.x_vit_gat/        ← Versión experimental (F1 LB = 0.382)
│   └── (mismo layout que v1.9)
│
├── tests/                  ← Fixtures compartidas (7×11 lesiones reales)
├── checkpoints/            ← Pesos PanDerm (`panderm_ll_data6_*.pth`, ViT-L)
├── Dataset/                ← MILK10k (GroundTruth, Metadata, imágenes)
│
└── docs/                   ← Documentación transversal del TFM
    ├── Clasificador … Memoria Dic 2025 … .pdf   ← memoria oficial
    ├── TFM_Resumen_Ejecutivo.md
    ├── HARDWARE.md · COMPARISON.md · RESULTS.md
    ├── PanDerm.GAT.V.1.9.PEC3.md                ← overview v1.9 (autocontenido)
    ├── Vit-GAT_v3_12_1_M4_FINAL.md              ← overview v3.12.x (autocontenido)
    └── assets/
        ├── architecture_v1_9.png                (Figura 3.2 del PDF)
        └── architecture_v3_12.png               (Figura 9.1 del PDF)
```

## Comparativa rápida

| Métrica (OOF / LB) | v1.9 | v3.12.x |
|---|---|---|
| **F1 Macro OOF** | **0.5411** | 0.5276 |
| **F1 Macro Leaderboard** | **0.478** | 0.382 |
| Balanced Accuracy OOF | 0.5192 | 0.5061 |
| AUC Macro OOF | 0.8883 | 0.8578 |
| Gap LB-OOF | 11.7 % | 27.7 % |
| Ramas | 3 (derm + clin + GAT) | 2 (clin + GAT) |
| Fusión | DualCLSGate → BranchGate | DualBranchGate único |
| Pérdida | Focal Loss (γ=1.5) + MSE | ProxyNCA++ |
| Nodos GAT | 8 | 11 |
| Rango gates | [0.25, 0.75] | [0.10, 0.90] |

→ **Mejor Modelo en resultados del Leaderboard: v1.9** (mejor generalización y menor gap LB).

## Entorno objetivo (real)

El TFM se ejecutó en la siguiente workstation; el código y las
configuraciones por defecto están ajustados a ese perfil:

- **CPU:** AMD Ryzen 9 5950X (16C/32T)
- **GPU:** NVIDIA **RTX 5090 (Blackwell, 32 GB VRAM, SM 12.0)**
- **Stack:** Linux + CUDA ≥ 12.8 + PyTorch ≥ 2.7 (validado 2.9.0+cu130)
- **Precisión:** AMP en **BF16 nativo** (sin GradScaler)

Ver `docs/HARDWARE.md` para detalles, tiempos por época y ajustes por versión.

### Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# Opcional: dependencias de test
pip install -r requirements-dev.txt
```

El repositorio funciona también en GPUs con menor memoria (≥ 16 GB) y CPU, pero
se tendrá que reducir `batch_size` y desactivar BF16 si tu tarjeta es
pre-Ampere.

## Uso rápido (cada versión)

```bash
# Entrenamiento 5-fold
python -m v1_9_monet_gat.scripts.train \
    --config v1.9_monet_gat/configs/default.yaml \
    --folds 5

# Evaluación OOF
python -m v1_9_monet_gat.scripts.evaluate \
    --ckpt v1.9_monet_gat/outputs/checkpoints/fold{0..4}.pt \
    --out  v1.9_monet_gat/outputs/metrics_oof.json

# Inferencia sobre test (submission ISIC)
python -m v1_9_monet_gat.scripts.infer \
    --ckpt v1.9_monet_gat/outputs/checkpoints/ \
    --test data/test_metadata.csv \
    --out  v1.9_monet_gat/outputs/submission.csv
```


## Dataset

- **MILK10k (ISIC Challenge 2025)**
- 5 240 lesiones de entrenamiento (10 480 imágenes: clínica + dermatoscópica)
- 479 lesiones de test (evaluación ciega)
- Tras aumentos D4: **41 920 muestras** de entrenamiento, 8 384 de validación por fold
- **11 clases** con fuerte desbalance (BCC 48 % → MAL_OTH 0.17 %)

## Cómo citar

Si usas este trabajo, por favor cita la memoria del TFM y este repositorio.

- **Memoria publicada (Zenodo)** — DOI: [10.5281/zenodo.18630494](https://doi.org/10.5281/zenodo.18630494)
- **Descubribilidad OpenAIRE**: [registro](https://explore.openaire.eu/search/result?pid=10.5281%2Fzenodo.18630494)
- **Depósito institucional UOC (O2)**: [hdl.handle.net/10609/154321](https://hdl.handle.net/10609/154321)
- **ORCID del autor**: [0009-0000-5714-912X](https://orcid.org/0009-0000-5714-912X)

### Cita recomendada 

> Muñoz Plá, M. (2025). *Clasificador Multimodal Híbrido de Lesiones Cutáneas en el desafío ISIC MILK10k 2025*. Trabajo Final de Máster, Universitat Oberta de Catalunya (UOC). https://hdl.handle.net/10609/154321



### Cita del dataset (CC-BY-NC)

El dataset MILK10k se distribuye bajo CC-BY-NC y exige cita explícita:

> MILK study team. *MILK10k*. ISIC Archive, 2025. doi:[10.34970/648456](https://doi.org/10.34970/648456)

## Licencia del código

Código bajo **Apache License 2.0** (ver [`LICENSE`](LICENSE)).

### Dependencias con licencia propia

- **Dataset MILK10k (ISIC 2025)** — CC-BY-NC. Debe obtenerse y citarse por separado (ver sección anterior).
- **Checkpoint PanDerm** (`panderm_ll_data6_checkpoint-499.pth`) — sujeto a los términos del proyecto PanDerm original. No se redistribuye en este repositorio.
