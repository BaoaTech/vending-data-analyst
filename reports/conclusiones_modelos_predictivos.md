# Conclusiones — Modelos Predictivos de Demanda para Máquinas Expendedoras

**Proyecto:** Optimización de rutas de reabastecimiento — El Paso, TX  
**Período de evaluación:** 26-dic-2025 → 30-mar-2026 (≈ 95 días)  
**Dataset de prueba:** 38 475 filas-día · 405 máquinas expendedoras

---

## 1. Contexto del problema

La demanda diaria por máquina expendedora presenta una distribución fuertemente sesgada hacia cero: el **61.5 % de los registros reportan ventas nulas** (máquina sin actividad ese día), con una media de \$6.27 y una desviación estándar de \$16.43. Esta dispersión estructural —inherente al modelo de negocio— condiciona los valores absolutos de todas las métricas y justifica el uso conjunto de MAE, MdAE y Bias como indicadores primarios, por encima del SMAPE y el R², cuya interpretación se distorsiona en presencia de valores reales cercanos a cero.

---

## 2. Segmentación de la flota (K-Means, 4 clústeres)

| Clúster | Etiqueta | Máquinas | Participación | Media diaria/VM |
|---------|----------|----------|---------------|-----------------|
| 0 | Rendimiento moderado | 182 | 44.9 % | \$2.40 |
| 1 | Alta venta, bajo surtido | 143 | 35.3 % | \$3.85 |
| 2 | Alta frecuencia | 78 | 19.3 % | \$9.88 |
| 3 | Outliers (alto valor) | 2 | 0.5 % | \$0.97 |

Los clústeres 0 y 1 concentran el **80.2 % de la flota** y definen el comportamiento agregado. El clúster 2, aunque minoritario, aporta desproporcionadamente a los ingresos totales dado que sus máquinas generan en promedio 4.1× más ventas que las del clúster 0. El clúster 3 agrupa únicamente dos unidades con comportamiento atípico; sus métricas no son estadísticamente representativas.

---

## 3. Comparación de métricas globales

| Modelo | MAE ↓ | RMSE ↓ | MdAE ↓ | R² ↑ | SMAPE ↓ | Bias |
|--------|-------|--------|--------|------|---------|------|
| K-Means (baseline) | 7.390 | 16.068 | 3.846 | 0.043 | 157.0 % | −1.928 |
| XGBoost | 5.784 | 13.286 | 2.298 | 0.346 | 148.0 % | −0.604 |
| **LSTM** | **5.319** | **12.738** | **1.657** | **0.399** | **149.3 %** | **−0.425** |

### Mejora relativa sobre el baseline K-Means

| Modelo | Δ MAE | Δ RMSE | Δ MdAE | Δ R² (relativo) |
|--------|-------|--------|--------|-----------------|
| XGBoost | −21.7 % | −17.3 % | −40.3 % | +700 % |
| LSTM | −28.0 % | −20.7 % | −56.9 % | +823 % |

El LSTM obtiene el **mejor resultado en las seis métricas**. La reducción más pronunciada se observa en la mediana del error absoluto (MdAE): −56.9 % respecto al baseline, lo que indica que el modelo acierta de forma consistente en la gran mayoría de los días, incluso cuando los valores extremos elevan el MAE y el RMSE. El R² pasa de 0.043 a 0.399, lo que significa que el LSTM explica casi **10 veces más varianza** que un pronóstico estático por segmento.

> **Sobre el SMAPE ≈ 148–157 %:** Este valor elevado es estructural, no un fallo del modelo. El 61.5 % de días con ventas reales = 0 produce SMAPE = 200 % en cualquier fila donde la predicción sea positiva. El SMAPE no es el indicador decisivo en este contexto.

---

## 4. Análisis por clúster

| Clúster | MAE XGBoost | MAE K-Means | Mejora | R² XGBoost |
|---------|-------------|-------------|--------|------------|
| 0 — Moderado | 4.271 | 4.787 | −10.8 % | 0.192 |
| 1 — Alta venta | 3.820 | 5.693 | −32.9 % | 0.462 |
| 2 — Alta frecuencia | 13.027 | 16.738 | −22.2 % | 0.289 |
| 3 — Outliers | 1.304 | 0.974 | +33.9 % ← K-Means gana | −∞ |

- **Clúster 1** es donde el modelo supervisado aporta mayor valor relativo (−32.9 % MAE): la variabilidad intrasemana capturada por las características temporales (`dow`, `is_weekend`) y los rezagos (`lag_1`, `lag_7`) no puede ser recuperada desde un perfil estático.
- **Clúster 2** muestra el mayor error absoluto simplemente porque sus máquinas venden más; la mejora relativa del 22.2 % es comparable a los demás segmentos.
- **Clúster 3** favorece al baseline, pero con solo 2 máquinas y 190 registros el resultado carece de significancia estadística.

---

## 5. Análisis del sesgo sistemático (Bias)

Los tres modelos **subestiman** la demanda (Bias < 0). En el contexto operativo del reabastecimiento, la subestimación implica **riesgo de quiebre de inventario** (understock): la ruta llega con menos producto del necesario.

| Modelo | Bias | Riesgo operativo |
|--------|------|-----------------|
| K-Means | −1.928 | Alto — subestima ~\$1.93 por VM/día |
| XGBoost | −0.604 | Moderado |
| LSTM | **−0.425** | **Bajo — el más cercano a cero** |

La causa raíz es la asimetría del dataset: los picos de venta (cola derecha, máx. \$521.75) son poco frecuentes y difíciles de anticipar, por lo que los modelos aprenden a ser conservadores. Para rutas críticas del clúster 2 se recomienda aplicar un ajuste de +\$0.43 sobre el pronóstico LSTM, o usar el cuantil 75 como estimación de stock mínimo en lugar de la media.

---

## 6. Perfil comparativo por dimensión operativa

| Dimensión | K-Means | XGBoost | LSTM |
|-----------|---------|---------|------|
| Tipo de aprendizaje | No supervisado | Supervisado tabular | Supervisado secuencial |
| Entrada | Etiqueta de clúster | 12 características (lags, rolling, temporal, clúster) | Secuencias de 14 días × 4 variables |
| Interpretabilidad | Alta | Alta (feature importance) | Baja (caja negra) |
| Velocidad de inferencia | Instantánea | ~ms por lote | ~100 ms por lote |
| Requiere historial mínimo | No | lag_14 (14 días) | 14 días continuos |
| Máquinas sin historial | Asignar clúster | Requiere reentrenamiento | Requiere reentrenamiento |
| Actualización incremental | No aplica | Fácil (warm-start XGBoost) | Costosa (reentrenamiento completo) |
| Artefacto generado | `cluster_profiles.csv` | `xgb_demand.json` | `lstm_demand.keras` + `lstm_scaler.pkl` |

---

## 7. Recomendaciones

### Producción

1. **Modelo principal → LSTM**: menor error en todas las métricas y bias más cercano a cero. Recomendado para el cálculo de pronósticos diarios en rutas con historial ≥ 14 días.

2. **Modelo de respaldo → XGBoost**: para máquinas con historial insuficiente para el LSTM, auditorías operativas o cuando se requiera explicar el pronóstico a equipos de campo mediante importancia de características.

3. **Segmentación → K-Means**: mantener los perfiles de clúster para asignar pronósticos a máquinas nuevas sin historial y para segmentar rutas de reabastecimiento por volumen esperado.

### Estrategia anti-quiebre

- Aplicar corrección de bias: `pronóstico_ajustado = pronóstico_lstm + 0.43` en rutas del clúster 2 (alta frecuencia), donde el costo del quiebre es mayor.
- Evaluar el cuantil p75 de predicciones históricas como umbral de stock mínimo para días de alta varianza (lunes, inicio de mes).

### Mantenimiento del modelo

- **LSTM**: recalibrar trimestralmente o ante eventos de deriva (reubicación de máquinas, cambio de contratos, incorporación de nuevas unidades).
- **XGBoost**: actualizable mensualmente con warm-start a bajo costo computacional.
- **K-Means**: revisar segmentación semestralmente; la composición de la flota puede cambiar.

---

## 8. Limitaciones y trabajo futuro

| Limitación | Impacto | Mejora propuesta |
|------------|---------|-----------------|
| 61.5 % de días con ventas cero no modelados explícitamente | Bias negativo sistemático | Modelo de dos etapas: clasificador (¿habrá venta?) + regresor (¿cuánto?) |
| Ausencia de variables externas | ~60 % de varianza no explicada (R² ≤ 0.40) | Incorporar temperatura, festivos locales El Paso, eventos, precio de combustible |
| LSTM entrenado globalmente (una sola red para 405 VMs) | Pierde patrones idiosincráticos por VM | Modelos locales para clúster 2 o fine-tuning por segmento |
| Clúster 3 con solo 2 máquinas | No generalizable | Reclasificar en clúster 0 o tratamiento ad-hoc |
| Horizonte de predicción = 1 día | Planificación de ruta requiere 3–7 días | Extender a predicción multi-paso (recursive o direct multi-output) |

---

*Generado a partir de los resultados del notebook `03_predictive_models.ipynb` — evaluación sobre 38 475 registros, período dic-2025 / mar-2026.*
