# Inventario de Señales y Variables — Datasets BeatLabile

Referencia para el diseño de futuros estudios. Documenta **todas** las señales y
variables disponibles en los tres datasets, independientemente de las usadas en BeatLabile.

---

## 1. Clínic — UCI Quirúrgica, Hospital Clínic de Barcelona

**Monitor:** Philips Intellivue (MX700/800)  
**Formato:** `.vital` VitalRecorder  
**Estructura:**  `datasets/data/clinic/box<N>/<boxdir>/<patid>/<date>/<filename>.vital`  
**Contexto:** postoperatorio, UCI E043

### Señales waveform (tipo 1 — muestral continuo)

| Canal | fs (Hz) | Unidades | Descripción | Disponibilidad |
|-------|:-------:|----------|-------------|:--------------:|
| `Intellivue/ECG_II` | 500 | mV | ECG derivación II | ~40 % archivos |
| `Intellivue/ECG_III` | 500 | mV | ECG derivación III | ~95 % |
| `Intellivue/ECG_I` | 500 | mV | ECG derivación I | raro (<10 %) |
| `Intellivue/ABP` | 125 | mmHg | Presión arterial invasiva continua | ~40 % |
| `Intellivue/PLETH` | 125 | u.a. | Fotopletismografía (PPG) | ~100 % |

### Parámetros numéricos derivados del monitor (tipo 2 — ~1 Hz)

#### Cardiovascular
| Canal | Unidad | Descripción |
|-------|--------|-------------|
| `Intellivue/ECG_HR` | lpm | Frecuencia cardíaca (desde ECG) |
| `Intellivue/ECG_VPC_CNT` | n | Conteo de VPC (complejos ventriculares prematuros) |
| `Intellivue/ABP_SYS` | mmHg | Presión arterial sistólica invasiva |
| `Intellivue/ABP_DIA` | mmHg | Presión arterial diastólica invasiva |
| `Intellivue/ABP_MEAN` | mmHg | Presión arterial media invasiva |
| `Intellivue/ABP_HR` | lpm | FC derivada del ABP |
| `Intellivue/NIBP_SYS` | mmHg | PAS no invasiva (brazalete) |
| `Intellivue/NIBP_DIA` | mmHg | PAD no invasiva |
| `Intellivue/NIBP_MEAN` | mmHg | PAM no invasiva |
| `Intellivue/NIBP_HR` | lpm | FC derivada de NIBP |

#### Pulsioximetría
| Canal | Unidad | Descripción |
|-------|--------|-------------|
| `Intellivue/PLETH_SAT_O2` | % | SpO₂ |
| `Intellivue/PLETH_HR` | lpm | FC desde PPG |
| `Intellivue/PLETH_PERF_REL` | % | Índice de perfusión (amplitud relativa PPG) |

#### Segmento ST (6 derivaciones calculadas)
| Canal | Unidad | Descripción |
|-------|--------|-------------|
| `Intellivue/ST_I` | mV | ST derivación I |
| `Intellivue/ST_II` | mV | ST derivación II |
| `Intellivue/ST_III` | mV | ST derivación III |
| `Intellivue/ST_AVF` | mV | ST aVF |
| `Intellivue/ST_AVL` | mV | ST aVL |
| `Intellivue/ST_AVR` | mV | ST aVR |
| `Intellivue/ST_MCL` | mV | ST MCL1 (~75 %) |
| `Intellivue/ST_V` | mV | ST V (~75 %) |

#### Ventilación (cuando disponible, ~15 % de archivos)
| Canal | Unidad | Descripción |
|-------|--------|-------------|
| `Intellivue/RR` | rpm | Frecuencia respiratoria |
| `Intellivue/FIO2` | % | FiO₂ |
| `Intellivue/PEEP_CMH2O` | cmH₂O | PEEP |
| `Intellivue/PIP_CMH2O` | cmH₂O | Presión inspiratoria pico |
| `Intellivue/PPLAT_CMH2O` | cmH₂O | Presión meseta |
| `Intellivue/TV_EXP` | mL | Volumen tidal espiratorio |
| `Intellivue/TV_INSP` | mL | Volumen tidal inspiratorio |
| `Intellivue/VENT_RR` | rpm | FR del ventilador |

#### BIS (cuando disponible, ~10 % de archivos)
| Canal | Unidad | Descripción |
|-------|--------|-------------|
| `Intellivue/EEG_BIS` | 0–100 | Índice Biespectral (profundidad anestesia) |
| `Intellivue/EEG_BIS_ASYM` | % | Asimetría hemisférica BIS |
| `Intellivue/EEG_BIS_SQI` | % | Índice de calidad de señal BIS |
| `Intellivue/EEG_RATIO_SUPPRN` | % | Ratio de supresión EEG |
| `Intellivue/EMG_ELEC_POTL_MUSCL` | dB | Actividad EMG (artefacto muscular) |

### Variables clínicas disponibles

**Ninguna.** Solo waveforms. Sin acceso a demografía, diagnóstico, medicación ni laboratorios.

---

## 2. MIMIC-IV Waveform — UCI, BIDMC Boston

**Formato:** WFDB multi-segmento (`.hea` + `.dat`)  
**Estructura:** `datasets/data/mimic4wdb_full/waves/p<group>/p<subject_id>/<stay_id>/`  
**Contexto:** UCI médico-quirúrgica, Boston. DB pública PhysioNet.  
**Licencia:** Open Data Commons License v1.0

### Señales waveform (continuo muestral)

| Señal WFDB | fs (Hz) | Unidades | Descripción | Disponibilidad |
|------------|:-------:|----------|-------------|:--------------:|
| `II` | 250 | mV | ECG derivación II | ~96 % |
| `V` | 250 | mV | ECG derivación precordial V | ~96 % |
| `I` | 250 | mV | ECG derivación I | ~11 % |
| `III` | 250 | mV | ECG derivación III | ~28 % |
| `aVR` | 250 | mV | ECG aVR | ~72 % |
| `aVL` | 250 | mV | ECG aVL | ~6 % |
| `aVF` | 250 | mV | ECG aVF | ~6 % |
| `ECG #0` / `ECG #0+` / `ECG #0++` / `ECG #0+++` | 250 | mV | ECG lead genérico (sistema alternativo de etiquetado) | variable |
| `ABP` | 125 | mmHg | Presión arterial invasiva continua | ~4 % |
| `CVP` | 125 | mmHg | Presión venosa central | raro |
| `FAP` | 125 | mmHg | Presión arterial femoral invasiva | raro |
| `PAP` | 125 | mmHg | Presión arterial pulmonar invasiva | raro |
| `ICP` | 125 | mmHg | Presión intracraneal | raro |
| `Pleth` | 125 | u.a. | Fotopletismografía (SpO₂) | ~87 % |
| `Resp` | 62 | u.a. | Señal respiratoria (impedancia torácica) | ~98 % |

> **Nota importante:** `ABP` solo está disponible en ~4 % de los segmentos del subconjunto local
> (`mimic4wdb_full`). La mayoría de registros tienen ECG + Pleth + Resp, pero no ART invasiva.
> Los 150 casos seleccionados para BeatLabile fueron los que tenían ART + ECG simultáneamente.

### Variables clínicas disponibles

Las tablas clínicas de MIMIC-IV (diagnósticos ICD, medicación, laboratorios, notas) están en una
base de datos relacional separada disponible en PhysioNet bajo credenciales CITI + DUA firmado.
**No están integradas** en los archivos de waveform locales.

---

## 3. VitalDB — Perioperatorio, Hospital Universidad Nacional de Seúl

**Formato:** `.vital` VitalRecorder  
**Estructura:** `datasets/data/vitaldb/vital_full_cases/<caseid>.vital` (p. ej. `0007.vital`)  
**Contexto:** quirófano (intra + postop), Seúl, Corea del Sur.  
**Licencia:** CC BY 4.0. IRB H-1810-153-982. API pública: `https://api.vitaldb.net`

Los archivos contienen señales de **múltiples dispositivos** integrados en el mismo registro.

---

### 3.1 SNUADC — ADC del laboratorio SNU (señales waveform)

| Canal | fs (Hz) | Descripción | Disponibilidad |
|-------|:-------:|-------------|:--------------:|
| `SNUADC/ECG_II` | 500 | ECG derivación II | ~100 % |
| `SNUADC/ART` | 500 | Presión arterial invasiva (radial/femoral) | ~100 % |
| `SNUADC/PLETH` | 500 | Fotopletismografía (PPG) | ~87 % |
| `SNUADC/CVP` | 500 | Presión venosa central invasiva | ~47 % |
| `SNUADC/ECG_V5` | 500 | ECG derivación V5 | ~40 % |
| `SNUADC/FEM` | 500 | Presión arterial femoral | raro (~7 %) |

---

### 3.2 Solar8000 — Monitor Philips/GE Solar8000 (parámetros numéricos ~1 Hz)

#### Cardiovascular
| Canal | Unidad | Descripción | Disponibilidad |
|-------|--------|-------------|:--------------:|
| `Solar8000/ART_SBP` | mmHg | PAS invasiva | ~100 % |
| `Solar8000/ART_MBP` | mmHg | PAM invasiva | ~100 % |
| `Solar8000/ART_DBP` | mmHg | PAD invasiva | ~100 % |
| `Solar8000/HR` | lpm | Frecuencia cardíaca | ~100 % |
| `Solar8000/ST_II` | mV | ST derivación II | ~100 % |
| `Solar8000/ST_I` | mV | ST derivación I | ~40 % |
| `Solar8000/ST_III` | mV | ST derivación III | ~40 % |
| `Solar8000/ST_AVF` | mV | ST aVF | ~40 % |
| `Solar8000/ST_AVL` | mV | ST aVL | ~40 % |
| `Solar8000/ST_AVR` | mV | ST aVR | ~40 % |
| `Solar8000/ST_V5` | mV | ST V5 | ~40 % |
| `Solar8000/BT` | ºC | Temperatura corporal | ~93 % |
| `Solar8000/CVP` | mmHg | PVC numérica | ~53 % |
| `Solar8000/NIBP_SBP` | mmHg | PAS no invasiva | ~67 % |
| `Solar8000/NIBP_MBP` | mmHg | PAM no invasiva | ~67 % |
| `Solar8000/NIBP_DBP` | mmHg | PAD no invasiva | ~67 % |
| `Solar8000/FEM_SBP` | mmHg | PAS femoral | raro |
| `Solar8000/FEM_MBP` | mmHg | PAM femoral | raro |
| `Solar8000/FEM_DBP` | mmHg | PAD femoral | raro |
| `Solar8000/PA_SBP` | mmHg | Presión arteria pulmonar sistólica | raro |
| `Solar8000/PA_MBP` | mmHg | PAP media | raro |
| `Solar8000/PA_DBP` | mmHg | PAP diastólica | raro |

#### Pulsioximetría y gases
| Canal | Unidad | Descripción | Disponibilidad |
|-------|--------|-------------|:--------------:|
| `Solar8000/PLETH_SPO2` | % | SpO₂ | ~100 % |
| `Solar8000/PLETH_HR` | lpm | FC desde PPG | ~100 % |
| `Solar8000/ETCO2` | mmHg | CO₂ teleespiratorio | ~100 % |
| `Solar8000/INCO2` | mmHg | CO₂ inspiratorio | ~100 % |
| `Solar8000/FEO2` | % | O₂ espiratorio | ~100 % |
| `Solar8000/FIO2` | % | FiO₂ | ~100 % |
| `Solar8000/RR_CO2` | rpm | FR desde capnografía | ~100 % |
| `Solar8000/RR` | rpm | FR directa | raro |

#### Ventilación mecánica (parámetros del monitor)
| Canal | Unidad | Descripción | Disponibilidad |
|-------|--------|-------------|:--------------:|
| `Solar8000/VENT_RR` | rpm | FR fijada/real | ~100 % |
| `Solar8000/VENT_TV` | mL | Volumen tidal real | ~100 % |
| `Solar8000/VENT_MV` | L/min | Volumen minuto | ~100 % |
| `Solar8000/VENT_PIP` | cmH₂O | Presión inspiratoria pico | ~100 % |
| `Solar8000/VENT_PPLAT` | cmH₂O | Presión meseta | ~100 % |
| `Solar8000/VENT_MAWP` | cmH₂O | Presión media vías aéreas | ~100 % |
| `Solar8000/VENT_INSP_TM` | s | Tiempo inspiratorio | ~100 % |
| `Solar8000/VENT_SET_TV` | mL | Volumen tidal programado | ~60 % |
| `Solar8000/VENT_SET_PCP` | cmH₂O | Presión control programada | ~27 % |
| `Solar8000/VENT_SET_FIO2` | % | FiO₂ programada | ~13 % |

#### Agentes anestésicos (monitor de gases)
| Canal | Unidad | Descripción | Disponibilidad |
|-------|--------|-------------|:--------------:|
| `Solar8000/GAS2_AGENT` | — | Agente halogenado identificado | ~53 % |
| `Solar8000/GAS2_INSPIRED` | % | Concentración inspirada del agente | ~53 % |
| `Solar8000/GAS2_EXPIRED` | % | Concentración teleespiratoria del agente | ~53 % |

---

### 3.3 Primus — Máquina de anestesia Dräger Primus

#### Waveforms (continuo muestral)
| Canal | fs (Hz) | Descripción | Disponibilidad |
|-------|:-------:|-------------|:--------------:|
| `Primus/AWP` | 62.5 | Presión vías aéreas (curva real) | ~100 % |
| `Primus/CO2` | 62.5 | Capnografía (curva de CO₂ real) | ~100 % |

#### Parámetros numéricos (~1 Hz)
| Canal | Unidad | Descripción |
|-------|--------|-------------|
| `Primus/ETCO2` | % | CO₂ teleespiratorio |
| `Primus/INCO2` | % | CO₂ inspiratorio |
| `Primus/FIO2` | % | FiO₂ |
| `Primus/FEO2` | % | O₂ espiratorio |
| `Primus/FIN2O` | % | N₂O inspiratorio |
| `Primus/FEN2O` | % | N₂O espiratorio |
| `Primus/MAC` | — | MAC total (potencia anestésica) |
| `Primus/MV` | L/min | Volumen minuto |
| `Primus/TV` | mL | Volumen tidal |
| `Primus/RR_CO2` | rpm | FR desde capnografía |
| `Primus/PEEP_MBAR` | mbar | PEEP |
| `Primus/PIP_MBAR` | mbar | Presión inspiratoria pico |
| `Primus/PPLAT_MBAR` | mbar | Presión meseta |
| `Primus/MAWP_MBAR` | mbar | Presión media vías aéreas |
| `Primus/PAMB_MBAR` | mbar | Presión atmosférica ambiente |
| `Primus/COMPLIANCE` | mL/mbar | Compliancia pulmonar dinámica |
| `Primus/VENT_LEAK` | mL | Fuga del circuito ventilatorio |
| `Primus/FLOW_O2` | L/min | Flujo O₂ (fresh gas) |
| `Primus/FLOW_AIR` | L/min | Flujo aire (fresh gas) |
| `Primus/FLOW_N2O` | L/min | Flujo N₂O (fresh gas) |
| `Primus/INSP_DES` / `Primus/EXP_DES` | % | Desflurane inspirado/espirado (~40 %) |
| `Primus/INSP_SEVO` / `Primus/EXP_SEVO` | % | Sevoflurane inspirado/espirado (~47 %) |
| `Primus/SET_FIO2` | % | FiO₂ programada |
| `Primus/SET_TV_L` | L | Volumen tidal programado |
| `Primus/SET_RR_IPPV` | rpm | FR programada |
| `Primus/SET_PIP` | mbar | PIP programada |
| `Primus/SET_INSP_TM` | s | Tiempo inspiratorio programado |
| `Primus/SET_INSP_PRES` | mbar | Presión soporte inspiratorio (~27 %) |
| `Primus/SET_FRESH_FLOW` | L/min | Flujo total programado |
| `Primus/SET_INTER_PEEP` | mbar | PEEP programado |
| `Primus/SET_INSP_PAUSE` | % | Pausa inspiratoria programada |
| `Primus/SET_AGE` | años | Edad introducida en la máquina |

---

### 3.4 Orchestra — Bomba de infusión TCI (Fresenius Kabi)

Todos los canales a ~1 Hz. Modelo farmacocinético TCI (target-controlled infusion): CE = concentración
efecto, CP = concentración plasma, CT = objetivo programado.

| Fármaco | Canales | Notas | Disponibilidad |
|---------|---------|-------|:--------------:|
| **Propofol 20 mg/mL** | `PPF20_CE`, `PPF20_CP`, `PPF20_CT`, `PPF20_RATE` (mL/h), `PPF20_VOL` (mL) | Hipnótico EV | ~73 % |
| **Remifentanilo 20 µg/mL** | `RFTN20_CE`, `RFTN20_CP`, `RFTN20_CT`, `RFTN20_RATE`, `RFTN20_VOL` | Opioide EV | ~93 % |
| **Remifentanilo 50 µg/mL** | `RFTN50_*` igual estructura | Concentración alternativa | ~7 % |
| **Rocuronio** | `ROC_RATE`, `ROC_VOL` | Relajante muscular | ~13 % |
| **Furosemida** | `FUT_RATE`, `FUT_VOL` | Diurético | raro |
| **Norepinefrina** | `NEPI_RATE`, `NEPI_VOL` | Vasopresor | raro |
| **Fenitefrina** | `PHEN_RATE`, `PHEN_VOL` | Vasopresor | raro |
| **Prostaglandina E1** | `PGE1_RATE`, `PGE1_VOL` | Vasodilatador | raro |

---

### 3.5 BIS — Monitor de índice bispectral Medtronic (13/15 archivos, ~87 %)

| Canal | fs (Hz) | Unidad | Descripción |
|-------|:-------:|--------|-------------|
| `BIS/EEG1_WAV` | 128 | µV | Waveform EEG canal 1 (frontal izquierdo) |
| `BIS/EEG2_WAV` | 128 | µV | Waveform EEG canal 2 (frontal derecho) |
| `BIS/BIS` | ~1 | 0–100 | Índice bispectral (profundidad anestesia) |
| `BIS/EMG` | ~1 | dB | Actividad EMG (artefacto muscular) |
| `BIS/SEF` | ~1 | Hz | Frecuencia espectral límite 95 % |
| `BIS/SQI` | ~1 | % | Índice de calidad de señal |
| `BIS/SR` | ~1 | % | Suppression ratio (burst-suppression) |
| `BIS/TOTPOW` | ~1 | dB | Potencia EEG total |

---

### 3.6 EV1000 / ClearSight — Monitor hemodinámico Edwards (~27 % archivos)

| Canal | Unidad | Descripción |
|-------|--------|-------------|
| `EV1000/CO` | L/min | Gasto cardíaco |
| `EV1000/CI` | L/min/m² | Índice cardíaco |
| `EV1000/SV` | mL | Volumen sistólico |
| `EV1000/SVI` | mL/m² | Índice de volumen sistólico |
| `EV1000/SVV` | % | Variación del volumen sistólico |
| `EV1000/ART_MBP` | mmHg | PAM (del propio sensor) |
| `EV1000/CVP` | mmHg | PVC |
| `EV1000/SVR` | dyn·s/cm⁵ | Resistencia vascular sistémica |
| `EV1000/SVRI` | dyn·s/cm⁵/m² | Índice de RVS |

---

### 3.7 Vigileo / FloTrac — Monitor cardíaco Edwards (~13 % archivos)

| Canal | Unidad | Descripción |
|-------|--------|-------------|
| `Vigileo/CO` | L/min | Gasto cardíaco |
| `Vigileo/CI` | L/min/m² | Índice cardíaco |
| `Vigileo/SV` | mL | Volumen sistólico |
| `Vigileo/SVI` | mL/m² | Índice VS |
| `Vigileo/SVV` | % | Variación VS |

---

### 3.8 Vigilance — Monitor Swan-Ganz Edwards (raro, <7 %)

| Canal | Unidad | Descripción |
|-------|--------|-------------|
| `Vigilance/CO` | L/min | GC continuo (termodilución) |
| `Vigilance/CI` | L/min/m² | Índice cardíaco |
| `Vigilance/SVO2` | % | Saturación venosa mixta (O₂) |
| `Vigilance/BT_PA` | ºC | Temperatura en arteria pulmonar |
| `Vigilance/SQI` / `Vigilance/SNR` | — | Calidad/ruido de señal |

---

### 3.9 Variables clínicas — `cases.csv` (API VitalDB)

Un registro por caso. 6,388 casos en la DB pública completa.

#### Identificación y tiempos
| Variable | Tipo | Descripción |
|----------|------|-------------|
| `caseid` | int | ID único del caso |
| `subjectid` | int | ID del paciente (varios casos por paciente posible) |
| `casestart` / `caseend` | int (s) | Inicio/fin del registro, relativo a opstart |
| `anestart` / `aneend` | int (s) | Inicio/fin de anestesia |
| `opstart` / `opend` | int (s) | Inicio/fin de cirugía |
| `adm` / `dis` | int (s) | Ingreso/alta hospitalaria |

#### Demografía
| Variable | Tipo | Descripción |
|----------|------|-------------|
| `age` | float | Edad (años) |
| `sex` | str | Sexo (`M`/`F`) |
| `height` | float | Talla (cm) |
| `weight` | float | Peso (kg) |
| `bmi` | float | IMC (kg/m²) |

#### Clasificación y tipo de cirugía
| Variable | Tipo | Descripción |
|----------|------|-------------|
| `asa` | float | ASA physical status (1–5) |
| `emop` | int | Urgente (1) vs programada (0) |
| `department` | str | Servicio quirúrgico (p. ej. `General surgery`, `Thoracic`) |
| `optype` | str | Categoría de cirugía (p. ej. `Colorectal`, `Cardiac`) |
| `opname` | str | Nombre completo del procedimiento |
| `dx` | str | Diagnóstico operatorio (texto libre) |
| `approach` | str | Abordaje quirúrgico (`Open`, `Laparoscopic`, `Robotic`) |
| `position` | str | Posición del paciente (`Supine`, `Lithotomy`, `Prone`…) |
| `ane_type` | str | Tipo de anestesia (`General`, `Spinal`, `Epidural`, `Combined`) |

#### Vía aérea y accesos
| Variable | Tipo | Descripción |
|----------|------|-------------|
| `cormack` | str | Grado de Cormack-Lehane (`I`–`IV`) |
| `airway` | str | Tipo de vía aérea (`Oral`, `Nasal`) |
| `tubesize` | float | Talla del tubo orotraqueal (mm) |
| `dltubesize` | str | Talla del tubo doble lumen |
| `lmasize` | float | Talla de mascarilla laríngea |
| `iv1` / `iv2` | str | Localización acceso venoso periférico 1/2 |
| `aline1` / `aline2` | str | Localización línea arterial 1/2 |
| `cline1` / `cline2` | str | Localización catéter central 1/2 |

#### Comorbilidades preoperatorias
| Variable | Tipo | Descripción |
|----------|------|-------------|
| `preop_htn` | int | Hipertensión arterial previa (0/1) |
| `preop_dm` | int | Diabetes mellitus previa (0/1) |
| `preop_ecg` | str | ECG preoperatorio (texto: `Normal Sinus Rhythm`, etc.) |
| `preop_pft` | str | Prueba función pulmonar (texto: `Normal`, `Obstructive`…) |

#### Analíticas preoperatorias (en `cases.csv`)
| Variable | Unidad | Descripción |
|----------|--------|-------------|
| `preop_hb` | g/dL | Hemoglobina |
| `preop_plt` | ×10³/µL | Plaquetas |
| `preop_pt` | % | Tiempo protrombina (actividad %) |
| `preop_aptt` | s | APTT |
| `preop_na` | mEq/L | Sodio |
| `preop_k` | mEq/L | Potasio |
| `preop_gluc` | mg/dL | Glucosa |
| `preop_alb` | g/dL | Albúmina |
| `preop_ast` | U/L | AST |
| `preop_alt` | U/L | ALT |
| `preop_bun` | mg/dL | Urea nitrogenada (BUN) |
| `preop_cr` | mg/dL | Creatinina |
| `preop_ph` | — | pH arterial |
| `preop_hco3` | mEq/L | Bicarbonato |
| `preop_be` | mEq/L | Exceso de base |
| `preop_pao2` | mmHg | PaO₂ |
| `preop_paco2` | mmHg | PaCO₂ |
| `preop_sao2` | % | SaO₂ |

#### Registro intraoperatorio (`cases.csv`, variables `intraop_*`)
| Variable | Unidad | Descripción |
|----------|--------|-------------|
| `intraop_ebl` | mL | Sangrado estimado |
| `intraop_uo` | mL | Diuresis intraoperatoria |
| `intraop_rbc` | mL | Transfusión de concentrado de hematíes |
| `intraop_ffp` | mL | Plasma fresco congelado transfundido |
| `intraop_crystalloid` | mL | Cristaloides EV administrados |
| `intraop_colloid` | mL | Coloides EV administrados |
| `intraop_ppf` | mg | Propofol total (EV) |
| `intraop_mdz` | mg | Midazolam total |
| `intraop_ftn` | µg | Fentanilo total |
| `intraop_rocu` | mg | Rocuronio total |
| `intraop_vecu` | mg | Vecuronio total |
| `intraop_eph` | µg | Efedrina total |
| `intraop_phe` | µg | Fenilefrina total |
| `intraop_epi` | µg | Epinefrina total |
| `intraop_ca` | mg | Calcio EV total |

#### Outcomes postoperatorios
| Variable | Tipo | Descripción |
|----------|------|-------------|
| `death_inhosp` | int | Mortalidad intrahospitalaria (0/1) |
| `icu_days` | int | Días de gestión en UCI |

---

### 3.10 Variables de laboratorio — `labs.csv` (API VitalDB)

**Formato:** tabla larga (`caseid`, `dt`, `name`, `result`). 928,448 filas en la DB completa.
`dt` = tiempo relativo a `opstart` (s); valores negativos = analítica preoperatoria.

| Código `name` | Descripción | Unidades |
|---------------|-------------|---------|
| `alb` | Albúmina | g/dL |
| `alt` | Alanina aminotransferasa | U/L |
| `ammo` | Amonio | µg/dL |
| `aptt` | APTT | s |
| `ast` | Aspartato aminotransferasa | U/L |
| `be` | Exceso de base | mEq/L |
| `bun` | BUN (urea nitrogenada) | mg/dL |
| `ccr` | Aclaramiento creatinina | mL/min |
| `cl` | Cloro | mEq/L |
| `cr` | Creatinina | mg/dL |
| `crp` | Proteína C reactiva | mg/dL |
| `esr` | Velocidad de sedimentación | mm/h |
| `fib` | Fibrinógeno | mg/dL |
| `gfr` | Tasa filtración glomerular estimada | mL/min/1.73m² |
| `gluc` | Glucosa | mg/dL |
| `hb` | Hemoglobina | g/dL |
| `hco3` | Bicarbonato | mEq/L |
| `hct` | Hematocrito | % |
| `ica` | Calcio iónico | mEq/L |
| `k` | Potasio | mEq/L |
| `lac` | Lactato | mEq/L |
| `na` | Sodio | mEq/L |
| `p` | Fósforo | mg/dL |
| `pco2` | pCO₂ | mmHg |
| `ph` | pH | — |
| `plt` | Plaquetas | ×10³/µL |
| `po2` | pO₂ | mmHg |
| `pt%` | Tiempo protrombina (% actividad) | % |
| `ptinr` | INR | — |
| `ptsec` | Tiempo protrombina (segundos) | s |
| `sao2` | SaO₂ | % |
| `tbil` | Bilirrubina total | mg/dL |
| `tprot` | Proteínas totales | g/dL |
| `wbc` | Leucocitos | ×10³/µL |

---

## 4. Resumen Comparativo de Señales

| Señal | Clínic | MIMIC-IV | VitalDB |
|-------|:------:|:--------:|:-------:|
| ECG (waveform) | ✅ 500 Hz | ✅ 250 Hz | ✅ 500 Hz |
| ART invasiva (waveform) | ✅ 125 Hz (~40%) | ✅ 125 Hz (~4%) | ✅ 500 Hz |
| PPG/Pleth (waveform) | ✅ 125 Hz | ✅ 125 Hz | ✅ 500 Hz |
| Capnografía (waveform) | ❌ | ❌ | ✅ 62.5 Hz (Primus) |
| Presión vías aéreas (waveform) | ❌ | ❌ | ✅ 62.5 Hz (Primus) |
| EEG waveform | solo BIS (~10%) | ❌ | ✅ 128 Hz (~87%) |
| Señal respiratoria | ❌ | ✅ 62 Hz | ❌ numérico solo |
| CVP (waveform) | ❌ | ✅ 125 Hz (raro) | ✅ 500 Hz (~47%) |
| PAP invasiva | ❌ | ✅ 125 Hz (raro) | ✅ numérico (raro) |
| ICP | ❌ | ✅ 125 Hz (raro) | ❌ |
| SpO₂ numérico | ✅ | ✅ | ✅ |
| EtCO₂ numérico | ✅ (~15%) | ❌ | ✅ |
| Temperatura | ❌ | ❌ | ✅ |
| Parámetros ventilador | ✅ (~15%) | ❌ | ✅ |
| Gasto cardíaco / SV / SVV | ❌ | ❌ | ✅ (~27-40%) |
| Fármacos TCI (FC/concentración) | ❌ | ❌ | ✅ Orchestra |
| ST segment (leads) | ✅ (6-8 deriv.) | ❌ | ✅ (7 deriv.) |
| Demografía | ❌ | ⚠️ separado (CITI) | ✅ |
| Cirugía / ASA / comorbilidades | ❌ | ⚠️ separado (CITI) | ✅ |
| Analíticas preop | ❌ | ⚠️ separado (CITI) | ✅ (34 analitos) |
| Medicación intraop con dosis | ❌ | ⚠️ separado (CITI) | ✅ (15 fármacos) |
| Outcomes (mortalidad, UCI) | ❌ | ⚠️ separado (CITI) | ✅ |
