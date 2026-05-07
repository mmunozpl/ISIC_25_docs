# Clasificador Multimodal Híbrido ViT-GAT - Resumen Ejecutivo
## Memoria TFM Diciembre 2025 - Manuel Muñoz Plá

---

## 1. RESULTADOS FINALES Y MÉTRICAS

### 1.1 Validación Out-of-Fold (OOF) - Validación Interna

| Métrica | Valor |
|---------|-------|
| **F1 Macro (principal)** | **0.5411** |
| Balanced Accuracy | 0.5192 |
| Accuracy Global | 0.7600 |
| F1 Weighted | 0.7537 |
| Precision Macro | 0.5825 |
| Recall Macro | 0.5192 |
| Cohen's Kappa | 0.6595 |
| Matthews Correlation Coefficient | 0.6606 |
| AUC Macro (one-vs-rest) | 0.8883 |
| Top-2 Accuracy | 0.8954 |
| Top-3 Accuracy | 0.9431 |

**Nota:** Los valores de Cohen's Kappa y MCC indican "concordancia sustancial" según Landis y Koch.

### 1.2 Leaderboard ISIC MILK10k 2025 (Evaluación Ciega)

| Métrica | OOF | Leaderboard | Degradación |
|---------|-----|------------|------------|
| **F1 Macro** | 0.5411 | **0.478** | -0.063 (11.7%) |
| Balanced Accuracy | 0.5192 | 0.486 | -0.033 (0.6%) |

**Posición:** 15 de 50 equipos en el Challenge ISIC MILK10k 2025

### 1.3 Resultados por Clase (OOF)

| Clase | Soporte | Recall | Especificidad | Precisión | F1 | AUC |
|-------|---------|--------|---------------|-----------|-----|-----|
| BCC | 20,176 | 0.922 | 0.861 | 0.860 | 0.890 | 0.950 |
| NV | 5,968 | 0.725 | 0.972 | 0.812 | 0.766 | 0.962 |
| BKL | 4,352 | 0.460 | 0.960 | 0.569 | 0.509 | 0.858 |
| SCCKA | 3,784 | 0.678 | 0.967 | 0.673 | 0.675 | 0.946 |
| MEL | 3,600 | 0.684 | 0.956 | 0.592 | 0.635 | 0.945 |
| AKIEC | 2,424 | 0.509 | 0.972 | 0.524 | 0.517 | 0.924 |
| DF | 416 | 0.413 | 0.997 | 0.593 | 0.487 | 0.885 |
| INF | 400 | 0.297 | 0.997 | 0.509 | 0.375 | - |
| VASC | 376 | 0.723 | 0.997 | 0.659 | 0.689 | - |
| BEN_OTH | 352 | 0.310 | 0.998 | 0.612 | 0.411 | - |
| MAL_OTH | 72 | 0.000 | 0.999 | 0.000 | 0.000 | >0.5 |

**Patrones observados:**
- Fortaleza en clases mayoritarias: BCC (recall = 0.922), NV (recall = 0.725)
- Debilidad extrema en clases minoritarias: MAL_OTH (9 muestras originales, 72 OOF) sin predicciones correctas
- Contraste entre AUC (0.888) y F1 Macro (0.541) → modelo ordena bien probabilidades pero no calibra óptimamente

### 1.4 Estabilidad entre Pliegues

| Fold | F1 Macro | Época Óptima |
|------|----------|--------------|
| 0 | 0.5612 | 36 |
| 1 | 0.5518 | 22 |
| 2 | 0.5417 | 14 |
| 3 | 0.5340 | 18 |
| 4 | 0.5597 | 24 |
| **Media ± Std** | **0.5497 ± 0.0117** | — |

Desviación estándar muy baja (0.0117) → **generalización consistente entre pliegues**

---

## 2. VERSIÓN RECOMENDADA Y MODELO FINAL

### 2.1 Modelo Ganador: ViT-GAT v1.9

**La versión v1.9 es la recomendada** como modelo final del proyecto, presentada en el Challenge ISIC MILK10k 2025.

#### Comparación v1.9 vs v3.12.x

| Aspecto | v1.9 | v3.12.x | Resultado |
|--------|------|---------|-----------|
| **F1 OOF** | **0.5411** | 0.5276 | ✓ v1.9 (+0.014) |
| **F1 Leaderboard** | **0.478** | 0.382 | ✓ v1.9 (+0.096) |
| F1 Leaderboard Gap % | **11.7%** | 27.7% | ✓ v1.9 mejor generalización |
| Balanced Accuracy OOF | 0.5192 | 0.5061 | ✓ v1.9 |
| AUC OOF | 0.8883 | 0.8578 | ✓ v1.9 |

**Conclusión:** A pesar de que v3.12.x implementa técnicas más sofisticadas (ProxyNCA++, unfreeze gradual, nodos de artefactos), **v1.9 generaliza significativamente mejor**, con gap validación-test 43% menor (11.7% vs 27.7%).

### 2.2 Razón del Mejor Rendimiento de v1.9

La hipótesis principal es que:
1. **Focal Loss (ϱ=1.5)** es más efectiva que ProxyNCA++ para este dataset
2. **DualCLSGate con BranchGate**: fusión más estable y balanceada
3. **3 ramas (derm, clin, GAT)** vs 2 ramas en v3.12.x
4. **Rango de gates [0.25, 0.75]** más conservador que [0.10, 0.90]

---

## 3. ARQUITECTURA DETALLADA DEL MODELO FINAL (v1.9)

### 3.1 Componentes Principales

**Nombre:** Clasificador Multimodal Híbrido ViT-GAT v1.9

#### Backbone Visual
- **Modelo:** Vision Transformer (ViT-L/16) con pesos PanDerm
- **Inicialización:** Pesos preentrenados en corpus dermatológico de gran escala (20x mayor que MONET)
- **Parámetros totales:** 304M
- **Parámetros entrenables:** ≈3.4M
- **Entrada:** 224×224 píxeles
- **Salida:** Token [CLS] + 196 patch tokens (14×14)

#### Rama Visual (Dual-Modal)
1. **DualCLSGate:** Fusiona tokens [CLS] de imagen clínica y dermatoscópica
   - Compuerta aprendida ε que preferencia modalidad dermatoscópica
   - Fórmula: `h_fused = ε·h_derm + (1-ε)·h_clin`

2. **BranchGate:** Equilibra rama visual con rama relacional
   - Rango de compuerta: [0.25, 0.75]
   - Comportamiento observado: ~50/50 entre visual y GAT

#### Rama Relacional (Graph Attention Networks)
- **Nodos del grafo:** 8 nodos
  - 4 nodos diagnósticos (MONET: ulceración, pelo, vasculatura, eritema)
  - 4 nodos demográficos (edad, sexo, tono de piel, localización)
- **Arquitectura:** Graph Attention Network (GAT)
- **Entrada:** Mapas de activación de 7 Concept Heads supervisadas débilmente
- **Función:** Capturar relaciones espaciales y semánticas entre características clínicas

#### Mecanismos de Fusión
1. **DualCLSGate:** Modalidad visual
2. **BranchGate:** Combinación visual-relacional
3. **Artifact Gate:** Modulación de artefactos (no en v1.9, solo en v3.12.x)

### 3.2 Configuración de Entrenamiento

| Parámetro | Valor |
|-----------|-------|
| **Backbone** | ViT-L/16 (PanDerm) |
| **Resolución entrada** | 224×224 |
| **Batch size** | 128 |
| **Learning rate backbone** | 2e-6 (con scheduler OneCycleLR) |
| **Learning rate GAT/conceptos/head** | 1e-4 |
| **Scheduler** | OneCycleLR |
| **Épocas máximas** | 40 |
| **Warmup** | 8 épocas |
| **Early stopping paciencia** | 12 épocas |
| **Dropout (head/GAT)** | 0.6 / 0.5 |
| **Label smoothing** | 0.1 |
| **Pérdida principal** | Focal Loss (ϱ=1.5) + MSE conceptual |
| **Peso concepto** | 0.25 |
| **Optimizador** | AdamW |

### 3.3 Esquema de Entrenamiento Progresivo

**Fase 1 (Épocas 1-7):** Warmup completo
- Backbone congelado
- Compuertas fijadas en 50/50

**Fase 2 (Épocas 8-10):** Descongelado inicial
- Último bloque del backbone descongela
- Compuertas aún en 50/50

**Fase 3 (Épocas 11-17):** Liberación de compuertas
- Aprendizaje de ε y ϖ en rango [0.25, 0.75]
- Backbone aún parcialmente congelado

**Fase 4 (Épocas 18+):** Fine-tuning progresivo
- Descongelado gradual del backbone
- 2 bloques en época 18, 3 bloques en época 30

**Técnicas adicionales:**
- Precisión mixta automática (AMP)
- Aumento D4 (8 transformaciones geométricas)
- Validación cruzada estratificada 5-fold por lesión

### 3.4 Datos de Entrenamiento

- **Dataset:** MILK10k (ISIC Challenge 2025)
- **Lesiones entrenamiento:** 5,240 (10,480 imágenes: clínica + dermatoscópica)
- **Lesiones test:** 479 (evaluación ciega)
- **Muestras tras aumento D4:** 41,920 (entrenamiento) + 8,384 (validación por fold)
- **Clases:** 11 diagnósticos con fuerte desbalance (BCC = 48%, MAL_OTH = 0.17%)

---

## 4. TABLA COMPARATIVA DE VERSIONES

| Característica | v1.9 (Final) | v3.12.x |
|----------------|------------|---------|
| **Mecanismo fusión principal** | DualCLSGate → BranchGate | DualBranchGate único |
| **Ramas** | 3 (derm, clin, GAT) | 2 (clin, GAT) |
| **Pérdida principal** | Focal Loss (ϱ=1.5) | ProxyNCA++ |
| **Pérdida auxiliar** | — | ProxyNCA++ (3→) |
| **Rango gates** | [0.25, 0.75] | [0.10, 0.90] |
| **Encoder** | PanDerm (frozen) | PanDerm (unfreeze gradual) |
| **Nodos GAT** | 8 (4 diag + 4 demo) | 11 (4 diag + 4 demo + 3 artifact) |
| **F1 OOF** | 0.5411 | 0.5276 |
| **F1 Leaderboard** | 0.478 | 0.382 |
| **Gap %** | 11.7% | 27.7% |
| **Comportamiento gates** | Preferencia derm (derm), Equilibrio (visual-GAT 50/50) | Preferencia GAT (70% GAT) |

---

## 5. MENCIONES DE NOTEBOOKS Y CHECKPOINTS

### 5.1 Notebooks Utilizados

Documentación en Jupyter Notebooks versionados localmente:
- Desarrollo integral en entorno local con JupyterLab
- Trazabilidad completa desde preprocesado hasta evaluación
- Registro sistemático de hiperparámetros, configuraciones y métricas intermedias

### 5.2 Checkpoints del Modelo

**Configuración de almacenamiento:**
- `CKPT_DIR: ./output/model_output/checkpoints`
- **Checkpoints activos:** Sí (guardados en entrenamiento)
- Modelo final seleccionado: ViT-GAT v1.9 (época óptima media: 22.8)

---

## 6. ANÁLISIS DE DEGRADACIÓN Y FACTORES

### 6.1 Gap Validación-Leaderboard (11.7%)

**Causas identificadas:**

1. **Distribution Shift (Desplazamiento de Distribución):**
   - Cambio de prevalencia: proporción de clases en test puede diferir del entrenamiento
   - Cambio covariante: dispositivos, iluminación, protocolos clínicos distintos
   - BCC experimenta degradación crítica (Δrecall = -0.47 en leaderboard)
   - Nuevos diagnósticos ISIC-DX granulares en test

2. **Overfitting Parcial:**
   - Gap entrenamiento-validación: F1_train ≈ 0.97 vs F1_val ≈ 0.54
   - Arquitectura compleja (304M parámetros) puede capturar patrones específicos
   - Regularización (dropout, label smoothing, aumento D4) potencialmente insuficiente

3. **Calibración de Temperatura:**
   - Temperatura softmax T=0.10 mejora F1 leaderboard en +0.011
   - Menor entropía favorece decisiones más firmes

### 6.2 Degradación por Clase

| Clase | Recall OOF | Recall Leaderboard | Δ | Valoración |
|-------|-----------|------------------|-----|-----------|
| BCC | 0.922 | 0.452 | -0.470 | **Crítica** |
| NV | 0.725 | 0.603 | -0.122 | Moderada |
| MEL | 0.684 | 0.600 | -0.084 | Moderada |
| SCCKA | 0.678 | 0.566 | -0.112 | Moderada |
| AKIEC | 0.509 | 0.411 | -0.098 | Moderada |
| [Otros] | Variable | Variable | Variable | Moderada-Leve |

---

## 7. RECOMENDACIONES Y CONCLUSIONES

### 7.1 Modelo Recomendado

✅ **ViT-GAT v1.9** es el modelo recomendado para producción y validación clínica futura.

**Justificación:**
- F1 Macro leaderboard: **0.478** (vs 0.382 en v3.12.x)
- Menor gap validación-test: **11.7%** (vs 27.7% en v3.12.x)
- Estabilidad entre pliegues: desviación estándar de 0.0117
- Recall competitivo en clases críticas: BCC (0.922), MEL (0.684)
- Posición respectable: **15/50 equipos** en challenge de nivel investigación

### 7.2 Puntos Fuertes

1. **AUC elevado (0.888):** Modelo ordena bien probabilidades
2. **Top-3 Accuracy (0.943):** Clase correcta en top-3 en 94.3% de casos
3. **Recall robusto en malignos frecuentes:** BCC = 0.922, MEL = 0.684
4. **Arquitectura multimodal justificada:** Integración efectiva de clínica + dermatoscopia + conceptos
5. **Metodología rigurosa:** Validación cruzada estratificada por lesión, sin fuga

### 7.3 Limitaciones Críticas

1. **Clases minoritarias:** MAL_OTH sin predicciones correctas (n=9 original)
2. **Gap significativo validación-test:** 11.7% degradación sugiere distribución distinta
3. **Dependencia de calibración:** Pequeñas variaciones arquitectónicas producen cambios apreciables
4. **Requisitos computacionales:** 304M parámetros requieren GPU para inferencia

### 7.4 Trabajos Futuros

1. **Calibración post-hoc:** Técnicas de ajuste de temperatura o Platt scaling
2. **Data augmentation avanzada:** Estrategias específicas para clases minoritarias
3. **Ensemble methods:** Combinación con modelos complementarios
4. **Validación clínica externa:** Evaluación prospectiva con casos nuevos
5. **Interpretabilidad:** Análisis de attention maps y concepto relevance

---

## 8. INFORMACIÓN TÉCNICA RESUMIDA

| Concepto | Valor |
|----------|-------|
| Autor | Manuel Muñoz Plá |
| Institución | UOC (Universitat Oberta de Catalunya) |
| Tipo Trabajo | TFM (Trabajo Fin de Máster) |
| Fecha | Diciembre 2025 |
| Challenge | ISIC MILK10k 2025 |
| Posición | 15 de 50 equipos |
| Métrica Principal | F1 Macro = 0.478 |
| Métrica OOF | F1 Macro = 0.5411 |
| Notebook Final | ViT-GAT v1.9 |
| Framework | PyTorch + Hugging Face Transformers |
| Dataset | MILK10k (5,240 lesiones entrenamiento) |

---

**Documentación generada:** 2025-04-23  
**Fuente:** Clasificador Multimodal Híbrido ViT-GAT Memoria TFM Dic 2025 - Manuel Muñoz Plá
