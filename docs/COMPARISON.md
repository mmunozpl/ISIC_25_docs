# Comparación ViT-GAT v1.9 ↔ v3.12.x

Datos extraídos directamente de la memoria TFM "Clasificador Multimodal
Híbrido ViT-GAT" (Diciembre 2025, Manuel Muñoz Plá).

## 1. Arquitectura

| Característica | **v1.9 (final)** | v3.12.x |
|---|---|---|
| Mecanismo de fusión principal | `DualCLSGate → BranchGate` | `DualBranchGate` único |
| Ramas | 3 (derm + clin + GAT) | 2 (clin + GAT) |
| Pérdida principal | Focal Loss (γ = 1.5) | ProxyNCA++ |
| Pérdida auxiliar | MSE conceptual (λ = 0.25) | CE aux (α = 0.5) + MSE concept / artifact |
| Rango gates | [0.25, 0.75] | [0.10, 0.90] |
| Encoder | PanDerm (unfreeze por fases 8/18/30) | PanDerm (unfreeze gradual 4→30) |
| Nodos GAT | 8 (4 diag + 4 demo) | 11 (4 diag + 4 demo + 3 artifact) |
| Comportamiento de gates | derm dominante; visual↔GAT ≈ 50 / 50 | GAT dominante ≈ 70 % |

## 2. Métricas

| Métrica | **v1.9** | v3.12.x | Δ (v1.9 − v3.12.x) |
|---|---|---|---|
| F1 Macro OOF | **0.5411** | 0.5276 | +0.014 |
| F1 Macro Leaderboard | **0.478** | 0.382 | +0.096 |
| Gap LB-OOF | **11.7 %** | 27.7 % | −16.0 p.p. |
| Balanced Accuracy OOF | **0.5192** | 0.5061 | +0.013 |
| AUC Macro OOF | **0.8883** | 0.8578 | +0.031 |
| Accuracy OOF | 0.7600 | (no reportado) | — |
| Top-3 Accuracy OOF | 0.9431 | (no reportado) | — |
| Cohen's κ OOF | 0.6595 | — | — |
| MCC OOF | 0.6606 | — | — |

## 3. Recomendación

**Modelo recomendado para producción / validación clínica: v1.9.**

Justificación (memoria §7.1):
- F1 Leaderboard 0.478 (vs 0.382) — **+25 % relativo**.
- Menor brecha validación ↔ test: 11.7 % (vs 27.7 %).
- Estabilidad entre pliegues σ = 0.0117.
- Recall competitivo en clases críticas (BCC 0.922, MEL 0.684).
- Posición final 15 / 50 equipos ISIC MILK10k 2025.

## 4. Por qué v1.9 generaliza mejor

1. **Focal Loss γ = 1.5** es más efectiva que ProxyNCA++ en este dataset
   (clases minoritarias < 100 muestras colapsan sus proxies).
2. **DualCLSGate + BranchGate** fusiona de forma **más estable** que
   DualBranchGate único (cuyo rango [0.10, 0.90] tiende a colapsos).
3. **Tres ramas** capturan información complementaria clínica ⊕ dermatoscópica.
4. **Gates [0.25, 0.75]** conservan información mutua entre modalidades.

## 5. Líneas de mejora (ambas versiones)

- Calibración post-hoc (ya se usa T = 0.10 → +0.011 F1 LB) y Platt scaling.
- Data augmentation específica para clases minoritarias (MAL_OTH, INF).
- Ensembles heterogéneos (v1.9 × v3.12.x con pesos por clase).
- Validación clínica externa con casos prospectivos.
- Interpretabilidad: attention maps y concept relevance por paciente.
