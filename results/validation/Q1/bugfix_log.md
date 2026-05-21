# Bugfix Log — Sub-estudio Q1 (señales ECG/PPG/PTT)

Fecha: 2025-06  
Archivos afectados: `validation/q1_main.py`, `validation/q1_signals.py`, `validation/q1_config.py`

---

## BUG 1 — PPG valid% inflado (9936.3% imposible)

### Síntoma
El informe final (`Q1_report.md`) mostraba valores como `9936.3%` y `7620.5%` para
`ppg_valid_pct` en lugar de los valores esperados (`99.4%`, `76.2%`).

### Root cause
En `q1_signals.py` (función `process_ppg`), la variable `pct_valid` se calculaba ya
en escala [0, 100]:

```python
pct_valid = float(valid_range.mean()) * 100   # → ya está en [0, 100]
```

Sin embargo, en **dos lugares** de `q1_main.py` se aplicaba una segunda multiplicación
por 100:

- Línea ~182 (warning de calidad): `ppg.pct_valid * 100 if ppg else 0.0`
- Línea ~358 (tabla del informe): `pct = ppg.pct_valid * 100 if ppg else 0.0`

El resultado era `99.4 × 100 = 9940`, produciendo el valor imposible.

### Impacto en exclusiones
**Ninguno.** La exclusión de `5684023` (PPG <70%) fue realizada por el script de
auditoría de cohorte (`scripts/audit_cohorte_hombro.py`), que usaba los valores
correctos calculados independientemente. La decisión de exclusión no cambia.

### Fix aplicado
Eliminadas las dos multiplicaciones redundantes en `q1_main.py`:

```python
# Antes (incorrecto):
ppg.pct_valid * 100 if ppg else 0.0

# Después (correcto):
ppg.pct_valid if ppg else 0.0
```

---

## BUG 2 — 70551555 solo 827 R-peaks en 97 min (~8 bpm, imposible)

### Síntoma
Paciente `70551555`: 827 R-peaks en 97 min → ~8.5 bpm. Fisiológicamente imposible.

### Diagnóstico
Script `diag_bugs.py` cargó las tres derivaciones disponibles para el paciente:

| Derivación         | Variabilidad (ptp) | R-peaks detectados | RR válidos |
|--------------------|-------------------:|-------------------:|-----------:|
| `Intellivue/ECG_II`  | ~0.002 (flat)      | 827                | ~827       |
| `Intellivue/ECG_V`   | ~0.001 (flat)      | muy pocos          | ~0         |
| `Intellivue/ECG_III` | ~0.8 mV (normal)   | **4663**           | **~4663**  |

Las derivaciones ECG_II y ECG_V son señales planas para este paciente (fallo de
electrodo). ECG_III es normal y da ~80 bpm (4663 peaks en 97 min).

### Root cause
La lista `ECG_TRACKS` en `q1_config.py` no incluía `Intellivue/ECG_III`. La función
`pick_track` seleccionaba el **primer** track disponible (`ECG_II`), sin verificar
si la señal tenía variabilidad suficiente.

### Fix aplicado

1. **`q1_config.py`** — añadido `"Intellivue/ECG_III"` a `ECG_TRACKS`:
   ```python
   ECG_TRACKS = ["Intellivue/ECG_II", "Intellivue/ECG_I", "Intellivue/ECG_III", "Intellivue/ECG_V"]
   ```

2. **`q1_signals.py`** — reescrita `process_ecg` con selección automática por calidad:
   - Añadidos 3 helpers: `_ecg_detect_rpeaks`, `_count_valid_rr`, `_is_ecg_signal_valid`
   - La función ahora itera **todas** las derivaciones candidatas, descarta señales flat
     (`ptp < 1e-4`), detecta R-peaks en cada una y selecciona la que tiene más
     intervalos RR fisiológicamente válidos.
   - Se emite un `log.warning` cuando la derivación seleccionada no es la primera de la lista.

### Impacto esperado
- `70551555`: ~827 → ~4663 R-peaks, derivación = `ECG_III`, ~80 bpm ✓
- El fichero `signals_cache.pkl` debe eliminarse para que el pipeline regenere:
  ```powershell
  Remove-Item results/validation/Q1/signals_cache.pkl
  ```

---

## BUG 3 — PTT matching rate 2–11% (esperado >50%)

### Síntoma
Para 16/18 pacientes, el ratio `n_ptt_pairs / n_rpeaks` era del 2–11%, con valores
fisiológicos imposiblemente bajos. Solo `4214722` y `4912692` tenían ratios del 50–61%.

### Hipótesis investigadas

| H  | Descripción                                              | Resultado       |
|----|----------------------------------------------------------|-----------------|
| H1 | Feet PPG = peaks, no mínimos diastólicos → offset ~ –T  | Descartada: nk usa mínimos correctamente |
| H2 | Ventana [150, 400ms] demasiado estrecha                  | **CONFIRMADA**  |
| H3 | Desalineación temporal ECG↔PPG (relojes distintos)       | Descartada      |

### Evidencia (figura)
`results/validation/Q1/figures/ptt_latency_distribution_per_patient.png`

El histograma muestra que para 13/18 pacientes, el **cluster dominante de latencias
R→foot se sitúa en ~430 ms**, 30 ms por encima del límite superior de la ventana
(400 ms). Solo los 2 pacientes con buenos ratios tienen latencias más cortas (~200–350 ms),
probablemente por diferencias individuales en tono vascular.

### Justificación fisiológica
En pacientes anestesiados con propofol + remifentanil, la vasodilatación periférica
aumenta el tiempo de tránsito de la onda de pulso. La ventana [150, 400ms] fue diseñada
para pacientes despiertos o con tono vascular normal. Una ventana [100, 600ms] es más
apropiada para anestesia general.

### Fix aplicado
`q1_config.py`:
```python
# Antes:
PTT_MIN_MS = 150.0
PTT_MAX_MS = 400.0

# Después:
PTT_MIN_MS = 100.0
PTT_MAX_MS = 600.0
```

### Impacto esperado
PTT ratio esperado >50% para la mayoría de los pacientes tras el fix.

---

## SANITY CHECK — Fix y ampliación

### Problema original
La función `check_detection_quality` verificaba el ratio de **feet PPG aceptados vs
detectados** (`n_feet_kept / n_feet_raw`), no el ratio de **PTT matched vs R-peaks**.
Un paciente con PPG perfecto pero PTT ratio = 0% pasaba el check sin problema.
Además, el umbral "≥3 pacientes" era demasiado permisivo.

### Fix aplicado (`q1_main.py`)

1. **Genera `detection_quality_per_patient.csv`** antes del PASO 3, con columnas:
   `patient_id`, `n_rpeaks`, `n_ppg_events`, `n_ptt_pairs`, `ratio_ptt_to_rpeaks`,
   `mean_ptt_ms`, `median_ptt_ms`, `ppg_valid_pct`, `ecg_lead_used`.

2. **Sanity check estricto por PTT ratio**:
   - Si **más de 3 pacientes** tienen `ratio_ptt_to_rpeaks < 0.20` → `SystemExit(2)`
   - Si 1–3 pacientes tienen ratio bajo → `log.warning` (aceptable individualmente)

3. El sanity check original de PPG se mantiene como advertencia secundaria (ya no
   detiene el pipeline).

---

## Notas de re-ejecución

Antes de re-ejecutar el pipeline, eliminar la caché de señales:
```powershell
$env:PYTHONIOENCODING = "utf-8"
cd c:\Users\Ramsés\Desktop\Proyectos\beatlabile\validation
Remove-Item results\validation\Q1\signals_cache.pkl -ErrorAction SilentlyContinue
python q1_main.py --step 2
```

Los archivos **no modificados** en esta sesión de bugfix:
- `validation/q1_stats.py` — sin cambios
- `validation/q1_features.py` — sin cambios
- `validation/q1_features_clinical.py` — sin cambios
