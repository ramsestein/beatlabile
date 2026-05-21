# Informe de auditoría — Cohorte hombro/bloqueo

_Generado: 2026-05-07 16:14:08_

## 1. Resumen ejecutivo

**Cohorte total analizada**: 21 carpetas de paciente.
**Archivos .vital encontrados**: 19 pacientes con ≥1 archivo; 2 sin ningún .vital.
**Calidad PPG**: 18 pacientes con ≥70% señal válida; 1 con calidad baja; 0 con error de carga.
**NIBP disponible**: 18 pacientes con >10 lecturas.
**Anotaciones**: 19 pacientes con eventos detectados en el .vital.
**Asignación de grupo**: 21/21 pacientes asignados (13 interescalénico / 8 supra+axilar).
**Pacientes con nombre de archivo ambiguo**: 7.
**Pacientes con ≥1 lectura MAP <55 mmHg**: 12.
**Grupos de grabaciones duplicadas (mismo dtstart)**: 0 (0 carpetas afectadas).

### Veredicto por pregunta pre-especificada

| Pregunta | Veredicto preliminar | Caveat principal |
|----------|---------------------|-----------------|
| **Q1** — Features autonómicas PPG-derived discriminan pre-hipotensión | **POSIBLE** | NIBP intermitente limita definición de "pre-hipotensión"; n=20 da potencia muy baja |
| **Q2** — Firma autonómica alrededor de estímulos dolorosos | **POSIBLE** | Depende de calidad de anotaciones en .vital; verificar timing exacto |
| **Q3** — Perfil hemodinámico interescalénico vs supra+axilar | **POSIBLE** | n pequeña; ambos grupos tienen bloqueo regional — diferencia de efecto simpaticolítico puede ser menor que bloqueo vs sin-bloqueo |

> ⚠️ **Caveats críticos**: (1) Sin línea arterial, las features de BeatLabile original no son computables — solo surrogates PPG. (2) NIBP intermitente impide definir "hipotensión sostenida" con el mismo criterio que BeatLabile; hay que redefinir operativamente. (3) n=21 ofrece potencia estadística muy limitada para análisis multivariante. (4) Todos los pacientes tienen algún tipo de bloqueo regional; la comparación es **interescalénico vs supra+axilar**, no bloqueo vs sin-bloqueo.

## 2. Inventario de archivos

| Paciente     | #vitals | Tamaño total | Archivo(s)                    | Nombre ambiguo | Grupo           | Notas                    |
| ------------ | ------- | ------------ | ----------------------------- | -------------- | --------------- | ------------------------ |
| 230393       | 1       | 5.9 MB       | 230393.vital                  | no             | supra+axilar    |                          |
| 397651       | 1       | 4.7 MB       | 397651.vital                  | no             | supra+axilar    |                          |
| 4214722      | 1       | 5.0 MB       | scjrke68a_250605_090011.vital | ⚠️ sí          | interescalenico |                          |
| 4234018 mal  | 0       | 0.0 B        | —                             | no             | interescalenico | Sin archivos .vital      |
| 4247699      | 1       | 5.4 MB       | 4247699.vital                 | no             | interescalenico |                          |
| 4912692      | 1       | 6.6 MB       | 4912692.vital                 | no             | supra+axilar    |                          |
| 5020549      | 1       | 3.1 MB       | 5020549.vital                 | no             | interescalenico |                          |
| 5362391      | 1       | 4.0 MB       | 5362391.vital                 | no             | interescalenico |                          |
| 5431482      | 1       | 3.7 MB       | 5431482.vital                 | no             | interescalenico |                          |
| 5582912      | 1       | 4.3 MB       | scjrke68a_250619_085513.vital | ⚠️ sí          | interescalenico |                          |
| 5589679      | 1       | 4.0 MB       | scjrke68a_250612_120310.vital | ⚠️ sí          | interescalenico |                          |
| 5684023      | 1       | 5.3 MB       | scjrke68a_250508_083922.vital | ⚠️ sí          | interescalenico |                          |
| 70078466     | 0       | 0.0 B        | —                             | no             | supra+axilar    | Sin archivos .vital      |
| 70288016     | 1       | 6.7 MB       | 70288016.vital                | no             | supra+axilar    |                          |
| 70297385     | 1       | 3.6 MB       | 70297385.vital                | no             | interescalenico |                          |
| 70431992     | 1       | 6.3 MB       | 70431992.vital                | no             | supra+axilar    |                          |
| 70436283     | 1       | 4.6 MB       | scjrke68a_250619_125648.vital | ⚠️ sí          | interescalenico |                          |
| 70551555     | 1       | 3.6 MB       | 70551555.vital                | no             | supra+axilar    |                          |
| 70628874     | 1       | 4.5 MB       | scjrke68a_250904_085330.vital | ⚠️ sí          | supra+axilar    |                          |
| 70767707 mal | 1       | 1.6 MB       | 70767707.vital                | no             | interescalenico | ⚠️ carpeta marcada 'mal' |
| 720142       | 1       | 2.5 MB       | scjrke68a_250612_101310.vital | ⚠️ sí          | interescalenico |                          |

**Total archivos .vital**: 19 | **Tamaño total**: 85.3 MB

**Pacientes SIN archivo .vital**: 4234018 mal, 70078466

**Pacientes con nombre de archivo ambiguo**: 4214722, 5582912, 5589679, 5684023, 70436283, 70628874, 720142

## 3. Inventario de señales

### 3.1 Señales esperadas — disponibilidad por paciente

| Señal esperada     | Pacientes con señal | % cohorte | Estado |
| ------------------ | ------------------- | --------- | ------ |
| PLETH/PPG (500 Hz) | 19/21               | 90%       | ⚠️     |
| NIBP_SBP           | 19/21               | 90%       | ⚠️     |
| NIBP_DBP           | 19/21               | 90%       | ⚠️     |
| NIBP_MAP/MBP       | 19/21               | 90%       | ⚠️     |
| HR continua        | 19/21               | 90%       | ⚠️     |
| SpO2               | 19/21               | 90%       | ⚠️     |
| BIS                | 16/21               | 76%       | ⚠️     |
| Resp/RR            | 3/21                | 14%       | ❌      |
| EtCO2              | 0/21                | 0%        | ❌      |

### 3.2 Tracks disponibles por paciente

**230393** — duración total: 85.9 min | tracks: 27

| Track                          | Tipo          | Srate  | Unidad | Min | Max |
| ------------------------------ | ------------- | ------ | ------ | --- | --- |
| EVENT                          | evento/string | evento |        | —   | —   |
| Intellivue/ECG_HR              | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ECG_I               | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_II              | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_III             | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_V               | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/EEG                 | onda continua | 125 Hz | uV     | —   | —   |
| Intellivue/EEG_BIS             | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/EEG_BIS_SQI         | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EEG_RATIO_SUPPRN    | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EMG_ELEC_POTL_MUSCL | numérico      | 1 Hz   | dB     | —   | —   |
| Intellivue/NIBP_DIA            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_HR             | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/NIBP_MEAN           | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_SYS            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/PLETH               | onda continua | 125 Hz |        | —   | —   |
| Intellivue/PLETH_HR            | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/PLETH_PERF_REL      | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/PLETH_SAT_O2        | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/ST_AVF              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVR              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_I                | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_II               | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_III              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_MCL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_V                | numérico      | 1 Hz   | mm     | —   | —   |

**397651** — duración total: 93.5 min | tracks: 26

| Track                          | Tipo          | Srate  | Unidad | Min | Max |
| ------------------------------ | ------------- | ------ | ------ | --- | --- |
| EVENT                          | evento/string | evento |        | —   | —   |
| Intellivue/ECG_HR              | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ECG_II              | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_III             | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_V               | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/EEG                 | onda continua | 125 Hz | uV     | —   | —   |
| Intellivue/EEG_BIS             | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/EEG_BIS_SQI         | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EEG_RATIO_SUPPRN    | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EMG_ELEC_POTL_MUSCL | numérico      | 1 Hz   | dB     | —   | —   |
| Intellivue/NIBP_DIA            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_HR             | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/NIBP_MEAN           | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_SYS            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/PLETH               | onda continua | 125 Hz |        | —   | —   |
| Intellivue/PLETH_HR            | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/PLETH_PERF_REL      | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/PLETH_SAT_O2        | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/ST_AVF              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVR              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_I                | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_II               | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_III              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_MCL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_V                | numérico      | 1 Hz   | mm     | —   | —   |

**4214722** — duración total: 78.4 min | tracks: 26

| Track                          | Tipo          | Srate  | Unidad | Min | Max |
| ------------------------------ | ------------- | ------ | ------ | --- | --- |
| EVENT                          | evento/string | evento |        | —   | —   |
| Intellivue/ECG_HR              | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ECG_II              | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_III             | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_V               | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/EEG                 | onda continua | 125 Hz | uV     | —   | —   |
| Intellivue/EEG_BIS             | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/EEG_BIS_SQI         | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EEG_RATIO_SUPPRN    | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EMG_ELEC_POTL_MUSCL | numérico      | 1 Hz   | dB     | —   | —   |
| Intellivue/NIBP_DIA            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_HR             | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/NIBP_MEAN           | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_SYS            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/PLETH               | onda continua | 125 Hz |        | —   | —   |
| Intellivue/PLETH_HR            | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/PLETH_PERF_REL      | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/PLETH_SAT_O2        | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/ST_AVF              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVR              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_I                | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_II               | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_III              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_MCL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_V                | numérico      | 1 Hz   | mm     | —   | —   |

**4234018 mal**: sin tracks (error de carga o sin .vital)

**4247699** — duración total: 84.9 min | tracks: 22

| Track                     | Tipo          | Srate  | Unidad | Min | Max |
| ------------------------- | ------------- | ------ | ------ | --- | --- |
| EVENT                     | evento/string | evento |        | —   | —   |
| Intellivue/ECG_HR         | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ECG_II         | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_III        | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_V          | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/NIBP_DIA       | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_HR        | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/NIBP_MEAN      | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_SYS       | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/PLETH          | onda continua | 125 Hz |        | —   | —   |
| Intellivue/PLETH_HR       | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/PLETH_PERF_REL | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/PLETH_SAT_O2   | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/RR             | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ST_AVF         | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVL         | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVR         | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_I           | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_II          | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_III         | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_MCL         | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_V           | numérico      | 1 Hz   | mm     | —   | —   |

**4912692** — duración total: 95.7 min | tracks: 26

| Track                          | Tipo          | Srate  | Unidad | Min | Max |
| ------------------------------ | ------------- | ------ | ------ | --- | --- |
| EVENT                          | evento/string | evento |        | —   | —   |
| Intellivue/ECG_HR              | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ECG_II              | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_III             | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_V               | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/EEG                 | onda continua | 125 Hz | uV     | —   | —   |
| Intellivue/EEG_BIS             | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/EEG_BIS_SQI         | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EEG_RATIO_SUPPRN    | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EMG_ELEC_POTL_MUSCL | numérico      | 1 Hz   | dB     | —   | —   |
| Intellivue/NIBP_DIA            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_HR             | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/NIBP_MEAN           | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_SYS            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/PLETH               | onda continua | 125 Hz |        | —   | —   |
| Intellivue/PLETH_HR            | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/PLETH_PERF_REL      | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/PLETH_SAT_O2        | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/ST_AVF              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVR              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_I                | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_II               | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_III              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_MCL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_V                | numérico      | 1 Hz   | mm     | —   | —   |

**5020549** — duración total: 83.4 min | tracks: 24

| Track                          | Tipo          | Srate  | Unidad | Min | Max |
| ------------------------------ | ------------- | ------ | ------ | --- | --- |
| EVENT                          | evento/string | evento |        | —   | —   |
| Intellivue/ECG_HR              | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ECG_II              | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_III             | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_V               | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/EEG                 | onda continua | 125 Hz | uV     | —   | —   |
| Intellivue/EEG_BIS             | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/EEG_BIS_SQI         | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EEG_RATIO_SUPPRN    | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EMG_ELEC_POTL_MUSCL | numérico      | 1 Hz   | dB     | —   | —   |
| Intellivue/NIBP_DIA            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_HR             | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/NIBP_MEAN           | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_SYS            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/PLETH               | onda continua | 125 Hz |        | —   | —   |
| Intellivue/PLETH_HR            | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/PLETH_PERF_REL      | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/PLETH_SAT_O2        | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/ST_AVF              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVR              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_I                | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_II               | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_III              | numérico      | 1 Hz   | mm     | —   | —   |

**5362391** — duración total: 54.1 min | tracks: 22

| Track                     | Tipo          | Srate  | Unidad | Min | Max |
| ------------------------- | ------------- | ------ | ------ | --- | --- |
| EVENT                     | evento/string | evento |        | —   | —   |
| Intellivue/ECG_HR         | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ECG_II         | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_III        | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_V          | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/NIBP_DIA       | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_HR        | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/NIBP_MEAN      | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_SYS       | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/PLETH          | onda continua | 125 Hz |        | —   | —   |
| Intellivue/PLETH_HR       | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/PLETH_PERF_REL | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/PLETH_SAT_O2   | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/RR             | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ST_AVF         | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVL         | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVR         | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_I           | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_II          | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_III         | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_MCL         | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_V           | numérico      | 1 Hz   | mm     | —   | —   |

**5431482** — duración total: 92.5 min | tracks: 24

| Track                          | Tipo          | Srate  | Unidad | Min | Max |
| ------------------------------ | ------------- | ------ | ------ | --- | --- |
| EVENT                          | evento/string | evento |        | —   | —   |
| Intellivue/ECG_HR              | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ECG_II              | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_III             | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_V               | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/EEG                 | onda continua | 125 Hz | uV     | —   | —   |
| Intellivue/EEG_BIS             | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/EEG_BIS_SQI         | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EEG_RATIO_SUPPRN    | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EMG_ELEC_POTL_MUSCL | numérico      | 1 Hz   | dB     | —   | —   |
| Intellivue/NIBP_DIA            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_HR             | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/NIBP_MEAN           | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_SYS            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/PLETH               | onda continua | 125 Hz |        | —   | —   |
| Intellivue/PLETH_HR            | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/PLETH_PERF_REL      | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/PLETH_SAT_O2        | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/ST_AVF              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVR              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_I                | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_II               | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_III              | numérico      | 1 Hz   | mm     | —   | —   |

**5582912** — duración total: 85.7 min | tracks: 17

| Track                          | Tipo          | Srate  | Unidad | Min | Max |
| ------------------------------ | ------------- | ------ | ------ | --- | --- |
| EVENT                          | evento/string | evento |        | —   | —   |
| Intellivue/ECG_HR              | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ECG_II              | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_V               | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/EEG                 | onda continua | 125 Hz | uV     | —   | —   |
| Intellivue/EEG_BIS             | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/EEG_BIS_SQI         | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EEG_RATIO_SUPPRN    | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EMG_ELEC_POTL_MUSCL | numérico      | 1 Hz   | dB     | —   | —   |
| Intellivue/NIBP_DIA            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_HR             | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/NIBP_MEAN           | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_SYS            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/PLETH               | onda continua | 125 Hz |        | —   | —   |
| Intellivue/PLETH_HR            | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/PLETH_PERF_REL      | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/PLETH_SAT_O2        | numérico      | 1 Hz   | %      | —   | —   |

**5589679** — duración total: 75.4 min | tracks: 17

| Track                          | Tipo          | Srate  | Unidad | Min | Max |
| ------------------------------ | ------------- | ------ | ------ | --- | --- |
| EVENT                          | evento/string | evento |        | —   | —   |
| Intellivue/ECG_HR              | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ECG_II              | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_V               | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/EEG                 | onda continua | 125 Hz | uV     | —   | —   |
| Intellivue/EEG_BIS             | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/EEG_BIS_SQI         | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EEG_RATIO_SUPPRN    | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EMG_ELEC_POTL_MUSCL | numérico      | 1 Hz   | dB     | —   | —   |
| Intellivue/NIBP_DIA            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_HR             | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/NIBP_MEAN           | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_SYS            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/PLETH               | onda continua | 125 Hz |        | —   | —   |
| Intellivue/PLETH_HR            | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/PLETH_PERF_REL      | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/PLETH_SAT_O2        | numérico      | 1 Hz   | %      | —   | —   |

**5684023** — duración total: 76.7 min | tracks: 22

| Track                     | Tipo          | Srate  | Unidad | Min | Max |
| ------------------------- | ------------- | ------ | ------ | --- | --- |
| EVENT                     | evento/string | evento |        | —   | —   |
| Intellivue/ECG_HR         | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ECG_II         | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_III        | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_V          | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/NIBP_DIA       | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_HR        | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/NIBP_MEAN      | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_SYS       | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/PLETH          | onda continua | 125 Hz |        | —   | —   |
| Intellivue/PLETH_HR       | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/PLETH_PERF_REL | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/PLETH_SAT_O2   | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/RR             | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ST_AVF         | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVL         | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVR         | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_I           | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_II          | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_III         | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_MCL         | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_V           | numérico      | 1 Hz   | mm     | —   | —   |

**70078466**: sin tracks (error de carga o sin .vital)

**70288016** — duración total: 94.2 min | tracks: 27

| Track                          | Tipo          | Srate  | Unidad | Min | Max |
| ------------------------------ | ------------- | ------ | ------ | --- | --- |
| EVENT                          | evento/string | evento |        | —   | —   |
| Intellivue/ECG_HR              | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ECG_I               | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_II              | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_III             | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_V               | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/EEG                 | onda continua | 125 Hz | uV     | —   | —   |
| Intellivue/EEG_BIS             | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/EEG_BIS_SQI         | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EEG_RATIO_SUPPRN    | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EMG_ELEC_POTL_MUSCL | numérico      | 1 Hz   | dB     | —   | —   |
| Intellivue/NIBP_DIA            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_HR             | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/NIBP_MEAN           | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_SYS            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/PLETH               | onda continua | 125 Hz |        | —   | —   |
| Intellivue/PLETH_HR            | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/PLETH_PERF_REL      | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/PLETH_SAT_O2        | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/ST_AVF              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVR              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_I                | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_II               | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_III              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_MCL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_V                | numérico      | 1 Hz   | mm     | —   | —   |

**70297385** — duración total: 64.3 min | tracks: 26

| Track                          | Tipo          | Srate  | Unidad | Min | Max |
| ------------------------------ | ------------- | ------ | ------ | --- | --- |
| EVENT                          | evento/string | evento |        | —   | —   |
| Intellivue/ECG_HR              | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ECG_II              | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_III             | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_V               | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/EEG                 | onda continua | 125 Hz | uV     | —   | —   |
| Intellivue/EEG_BIS             | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/EEG_BIS_SQI         | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EEG_RATIO_SUPPRN    | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EMG_ELEC_POTL_MUSCL | numérico      | 1 Hz   | dB     | —   | —   |
| Intellivue/NIBP_DIA            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_HR             | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/NIBP_MEAN           | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_SYS            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/PLETH               | onda continua | 125 Hz |        | —   | —   |
| Intellivue/PLETH_HR            | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/PLETH_PERF_REL      | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/PLETH_SAT_O2        | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/ST_AVF              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVR              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_I                | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_II               | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_III              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_MCL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_V                | numérico      | 1 Hz   | mm     | —   | —   |

**70431992** — duración total: 85.8 min | tracks: 26

| Track                          | Tipo          | Srate  | Unidad | Min | Max |
| ------------------------------ | ------------- | ------ | ------ | --- | --- |
| EVENT                          | evento/string | evento |        | —   | —   |
| Intellivue/ECG_HR              | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ECG_II              | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_III             | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_V               | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/EEG                 | onda continua | 125 Hz | uV     | —   | —   |
| Intellivue/EEG_BIS             | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/EEG_BIS_SQI         | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EEG_RATIO_SUPPRN    | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EMG_ELEC_POTL_MUSCL | numérico      | 1 Hz   | dB     | —   | —   |
| Intellivue/NIBP_DIA            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_HR             | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/NIBP_MEAN           | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_SYS            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/PLETH               | onda continua | 125 Hz |        | —   | —   |
| Intellivue/PLETH_HR            | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/PLETH_PERF_REL      | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/PLETH_SAT_O2        | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/ST_AVF              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVR              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_I                | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_II               | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_III              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_MCL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_V                | numérico      | 1 Hz   | mm     | —   | —   |

**70436283** — duración total: 65.5 min | tracks: 26

| Track                          | Tipo          | Srate  | Unidad | Min | Max |
| ------------------------------ | ------------- | ------ | ------ | --- | --- |
| EVENT                          | evento/string | evento |        | —   | —   |
| Intellivue/ECG_HR              | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ECG_II              | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_III             | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_V               | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/EEG                 | onda continua | 125 Hz | uV     | —   | —   |
| Intellivue/EEG_BIS             | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/EEG_BIS_SQI         | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EEG_RATIO_SUPPRN    | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EMG_ELEC_POTL_MUSCL | numérico      | 1 Hz   | dB     | —   | —   |
| Intellivue/NIBP_DIA            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_HR             | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/NIBP_MEAN           | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_SYS            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/PLETH               | onda continua | 125 Hz |        | —   | —   |
| Intellivue/PLETH_HR            | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/PLETH_PERF_REL      | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/PLETH_SAT_O2        | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/ST_AVF              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVR              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_I                | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_II               | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_III              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_MCL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_V                | numérico      | 1 Hz   | mm     | —   | —   |

**70551555** — duración total: 97.4 min | tracks: 26

| Track                          | Tipo          | Srate  | Unidad | Min | Max |
| ------------------------------ | ------------- | ------ | ------ | --- | --- |
| EVENT                          | evento/string | evento |        | —   | —   |
| Intellivue/ECG_HR              | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ECG_II              | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_III             | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_V               | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/EEG                 | onda continua | 125 Hz | uV     | —   | —   |
| Intellivue/EEG_BIS             | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/EEG_BIS_SQI         | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EEG_RATIO_SUPPRN    | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EMG_ELEC_POTL_MUSCL | numérico      | 1 Hz   | dB     | —   | —   |
| Intellivue/NIBP_DIA            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_HR             | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/NIBP_MEAN           | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_SYS            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/PLETH               | onda continua | 125 Hz |        | —   | —   |
| Intellivue/PLETH_HR            | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/PLETH_PERF_REL      | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/PLETH_SAT_O2        | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/ST_AVF              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVR              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_I                | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_II               | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_III              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_MCL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_V                | numérico      | 1 Hz   | mm     | —   | —   |

**70628874** — duración total: 70.3 min | tracks: 26

| Track                          | Tipo          | Srate  | Unidad | Min | Max |
| ------------------------------ | ------------- | ------ | ------ | --- | --- |
| EVENT                          | evento/string | evento |        | —   | —   |
| Intellivue/ECG_HR              | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ECG_II              | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_III             | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_V               | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/EEG                 | onda continua | 125 Hz | uV     | —   | —   |
| Intellivue/EEG_BIS             | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/EEG_BIS_SQI         | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EEG_RATIO_SUPPRN    | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EMG_ELEC_POTL_MUSCL | numérico      | 1 Hz   | dB     | —   | —   |
| Intellivue/NIBP_DIA            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_HR             | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/NIBP_MEAN           | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_SYS            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/PLETH               | onda continua | 125 Hz |        | —   | —   |
| Intellivue/PLETH_HR            | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/PLETH_PERF_REL      | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/PLETH_SAT_O2        | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/ST_AVF              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVR              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_I                | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_II               | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_III              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_MCL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_V                | numérico      | 1 Hz   | mm     | —   | —   |

**70767707 mal** — duración total: 22.8 min | tracks: 26

| Track                          | Tipo          | Srate  | Unidad | Min | Max |
| ------------------------------ | ------------- | ------ | ------ | --- | --- |
| EVENT                          | evento/string | evento |        | —   | —   |
| Intellivue/ECG_HR              | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ECG_II              | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_III             | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_V               | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/EEG                 | onda continua | 125 Hz | uV     | —   | —   |
| Intellivue/EEG_BIS             | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/EEG_BIS_SQI         | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EEG_RATIO_SUPPRN    | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EMG_ELEC_POTL_MUSCL | numérico      | 1 Hz   | dB     | —   | —   |
| Intellivue/NIBP_DIA            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_HR             | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/NIBP_MEAN           | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_SYS            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/PLETH               | onda continua | 125 Hz |        | —   | —   |
| Intellivue/PLETH_HR            | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/PLETH_PERF_REL      | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/PLETH_SAT_O2        | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/ST_AVF              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_AVR              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_I                | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_II               | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_III              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_MCL              | numérico      | 1 Hz   | mm     | —   | —   |
| Intellivue/ST_V                | numérico      | 1 Hz   | mm     | —   | —   |

**720142** — duración total: 73.9 min | tracks: 17

| Track                          | Tipo          | Srate  | Unidad | Min | Max |
| ------------------------------ | ------------- | ------ | ------ | --- | --- |
| EVENT                          | evento/string | evento |        | —   | —   |
| Intellivue/ECG_HR              | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/ECG_II              | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/ECG_V               | onda continua | 500 Hz | mV     | —   | —   |
| Intellivue/EEG                 | onda continua | 125 Hz | uV     | —   | —   |
| Intellivue/EEG_BIS             | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/EEG_BIS_SQI         | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EEG_RATIO_SUPPRN    | numérico      | 1 Hz   | %      | —   | —   |
| Intellivue/EMG_ELEC_POTL_MUSCL | numérico      | 1 Hz   | dB     | —   | —   |
| Intellivue/NIBP_DIA            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_HR             | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/NIBP_MEAN           | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/NIBP_SYS            | numérico      | 1 Hz   | mmHg   | —   | —   |
| Intellivue/PLETH               | onda continua | 125 Hz |        | —   | —   |
| Intellivue/PLETH_HR            | numérico      | 1 Hz   | /min   | —   | —   |
| Intellivue/PLETH_PERF_REL      | numérico      | 1 Hz   |        | —   | —   |
| Intellivue/PLETH_SAT_O2        | numérico      | 1 Hz   | %      | —   | —   |

## 4. Calidad de PPG

| Paciente     | Dur total (min) | % válido | % clipping | % plano | #gaps>5s | Total gap (min) | SNR mediana (dB) | Flag         |
| ------------ | --------------- | -------- | ---------- | ------- | -------- | --------------- | ---------------- | ------------ |
| 230393       | 85.9            | 99.4%    | 0.06%      | 0.27%   | 0        | 0.0             | 11.4             |              |
| 397651       | 93.5            | 76.2%    | 0.04%      | 4.68%   | 6        | 22.2            | 12.1             |              |
| 4214722      | 78.4            | 99.4%    | 0.03%      | 0.00%   | 0        | 0.0             | 12.6             |              |
| 4234018 mal  | —               | —        | —          | —       | —        | —               | —                | Sin análisis |
| 4247699      | 84.9            | 97.0%    | 0.12%      | 3.40%   | 1        | 2.2             | 14.0             |              |
| 4912692      | 95.7            | 99.6%    | 0.13%      | 2.66%   | 0        | 0.0             | 13.2             |              |
| 5020549      | 83.4            | 99.5%    | 0.03%      | 1.17%   | 0        | 0.0             | 12.3             |              |
| 5362391      | 54.1            | 98.9%    | 0.02%      | 8.09%   | 1        | 0.2             | 13.2             |              |
| 5431482      | 92.5            | 98.5%    | 0.03%      | 5.12%   | 4        | 1.0             | 13.8             |              |
| 5582912      | 85.7            | 99.6%    | 0.04%      | 0.20%   | 0        | 0.0             | 12.5             |              |
| 5589679      | 75.4            | 99.8%    | 0.13%      | 3.76%   | 0        | 0.0             | 13.3             |              |
| 5684023      | 76.7            | 56.2%    | 0.08%      | 4.95%   | 5        | 33.6            | 16.6             | ⚠️ EXCLUIR   |
| 70078466     | —               | —        | —          | —       | —        | —               | —                | Sin análisis |
| 70288016     | 94.2            | 99.6%    | 0.09%      | 0.14%   | 0        | 0.0             | 12.5             |              |
| 70297385     | 64.3            | 81.0%    | 0.16%      | 5.70%   | 6        | 12.1            | 13.3             |              |
| 70431992     | 85.8            | 99.5%    | 0.17%      | 0.62%   | 0        | 0.0             | 11.0             |              |
| 70436283     | 65.5            | 99.2%    | 0.04%      | 0.00%   | 1        | 0.1             | 13.7             |              |
| 70551555     | 97.4            | 88.3%    | 0.23%      | 2.71%   | 4        | 11.2            | 12.3             |              |
| 70628874     | 70.3            | 99.4%    | 0.24%      | 0.00%   | 0        | 0.0             | 12.7             |              |
| 70767707 mal | 22.8            | 99.3%    | 0.13%      | 1.47%   | 0        | 0.0             | 15.2             |              |
| 720142       | 73.9            | 99.8%    | 0.04%      | 3.75%   | 0        | 0.0             | 16.7             |              |

> Umbral de exclusión: <70% PPG válido.
> **Método SNR**: potencia en banda cardíaca (0.5–4 Hz, Butterworth 4º orden) / potencia residual, en ventanas de 30 s. Mediana sobre todas las ventanas válidas.

**Candidatos a exclusión por calidad PPG**: 5684023

> Ver figura `ppg_quality_histogram.png` en el directorio de resultados.

## 5. Caracterización de NIBP

### 5.1 Estadísticas por paciente

| Paciente     | #lecturas | Ciclo medio (min) | Ciclo mediana (min) | SBP media±SD | SBP rango | DBP media±SD | MAP media±SD | #outliers SBP | #outliers DBP | #MAP<55 |
| ------------ | --------- | ----------------- | ------------------- | ------------ | --------- | ------------ | ------------ | ------------- | ------------- | ------- |
| 230393       | 46        | 1.8               | 1.0                 | 113±17       | [75-157]  | 71±8         | 80±10        | 0             | 0             | 0       |
| 397651       | 28        | 3.3               | 2.9                 | 126±10       | [107-144] | 66±7         | 80±6         | 0             | 0             | 0       |
| 4214722      | 34        | 2.4               | 1.1                 | 118±19       | [87-159]  | 63±5         | 76±9         | 0             | 0             | 0       |
| 4234018 mal  | —         | —                 | —                   | —            | —         | —            | —            | —             | —             | —       |
| 4247699      | 35        | 2.3               | 1.6                 | 93±11        | [76-125]  | 46±7         | 57±7         | 0             | 0             | 8       |
| 4912692      | 41        | 2.3               | 1.4                 | 105±19       | [64-149]  | 61±12        | 71±12        | 0             | 0             | 2       |
| 5020549      | 34        | 2.1               | 1.0                 | 103±12       | [90-134]  | 46±7         | 58±8         | 0             | 0             | 7       |
| 5362391      | 19        | 2.7               | 2.2                 | 99±8         | [84-118]  | 40±9         | 53±7         | 0             | 1             | 8       |
| 5431482      | 36        | 2.3               | 1.3                 | 86±12        | [74-127]  | 47±6         | 55±7         | 0             | 0             | 11      |
| 5582912      | 41        | 2.0               | 1.7                 | 90±10        | [62-116]  | 50±9         | 60±9         | 0             | 1             | 4       |
| 5589679      | 40        | 1.7               | 1.1                 | 89±12        | [73-118]  | 39±14        | 50±12        | 0             | 0             | 18      |
| 5684023      | 19        | 2.7               | 2.2                 | 94±9         | [81-111]  | 44±6         | 55±6         | 0             | 0             | 6       |
| 70078466     | —         | —                 | —                   | —            | —         | —            | —            | —             | —             | —       |
| 70288016     | 44        | 2.1               | 1.3                 | 112±21       | [69-160]  | 67±13        | 77±14        | 0             | 0             | 1       |
| 70297385     | 22        | 2.8               | 1.7                 | 93±23        | [50-166]  | 51±13        | 61±15        | 1             | 0             | 5       |
| 70431992     | 41        | 2.1               | 1.1                 | 128±22       | [96-185]  | 82±19        | 92±18        | 0             | 1             | 0       |
| 70436283     | 24        | 2.6               | 1.5                 | 106±9        | [93-134]  | 53±10        | 64±8         | 0             | 0             | 0       |
| 70551555     | 44        | 2.2               | 1.3                 | 101±19       | [75-146]  | 60±10        | 69±11        | 0             | 0             | 0       |
| 70628874     | 44        | 1.6               | 2.5                 | 123±21       | [89-172]  | 75±6         | 85±9         | 0             | 0             | 0       |
| 70767707 mal | 9         | 2.7               | 2.6                 | 102±10       | [91-121]  | 51±14        | 64±14        | 0             | 0             | 2       |
| 720142       | 33        | 2.0               | 1.3                 | 116±17       | [91-160]  | 44±7         | 61±9         | 0             | 0             | 3       |

### 5.2 Criterio operativo para "hipotensión" con NIBP intermitente

Con NIBP cada ~5 min (estimado) no se puede verificar "sostenido ≥3 min" como en BeatLabile.
**Propuesta de criterio operativo**:
- **Caso A (laxo)**: ≥1 lectura MAP < 65 mmHg en un intervalo de 30 min.
- **Caso B (estricto)**: ≥2 lecturas consecutivas MAP < 55 mmHg (implica ≥~5-10 min si ciclo ~5 min).
- Reportar ambos criterios y analizar sensibilidad.
- Si el ciclo NIBP es >10 min, la definición de "episodio hipotensivo" es poco fiable y debe indicarse explícitamente como limitación.

## 6. Anotaciones (estímulos y medicación)

**Total de anotaciones encontradas**: 274 | **Únicas**: 91

### 6.1 Anotaciones de estímulos quirúrgicos

| Texto                                        | Frecuencia |
| -------------------------------------------- | ---------- |
| piel                                         | 29         |
| anclaje                                      | 19         |
| trocar                                       | 18         |
| sutura                                       | 10         |
| anclaje 2                                    | 8          |
| trocar 2                                     | 8          |
| trocar 3                                     | 5          |
| anclaje 3                                    | 5          |
| anclaje 4                                    | 4          |
| fresan                                       | 2          |
| ancle 2                                      | 1          |
| tiran la sutura                              | 1          |
| anclaje 5                                    | 1          |
| anclaje 6                                    | 1          |
| trocar

propo 2.8, remi 2

propo 2.5, remi 2 | 1          |
| sutura piel                                  | 1          |

### 6.2 Anotaciones de medicación

| Texto             | Frecuencia |
| ----------------- | ---------- |
| efedrina          | 17         |
| fenilefrina       | 3          |
| fentanilo 100 mcg | 1          |
| fentanilo 50 mcg  | 1          |
| fentanilo         | 1          |

### 6.3 Otras anotaciones (top 30)

| Texto               | Frecuencia |
| ------------------- | ---------- |
| AG                  | 18         |
| limpian             | 15         |
| fin AG              | 12         |
| silla playa         | 4          |
| propo 2, remi 2     | 4          |
| propo 3.5, remi 3   | 3          |
| propo 1.8, remi 1.5 | 3          |
| propo 2, remi 1.8   | 3          |
| propo 2.5, remi 2   | 2          |
| propo 2, remi 2.5   | 2          |
| DLI                 | 2          |
| propo 3, remi 3     | 2          |
| metadona 2mg        | 2          |
| propo 2, remi 1.5   | 2          |
| propo 1.6, remi 1.5 | 2          |
| remi 2              | 2          |
| propo 3, remi 2.5   | 2          |
| DLD                 | 2          |
| propo 3, remi 2     | 2          |
| fin IQ              | 2          |
| propo 2.6, remi 2   | 2          |
| propo 2.5, remi 2.5 | 1          |
| propo 2.5, remi 3   | 1          |
| propo 2.3, remi 3   | 1          |
| fenta               | 1          |
| propo 2, remi 2.8   | 1          |
| propo 2, remi 2.3   | 1          |
| propo 3, remi 3.5   | 1          |
| propo 3, remi 4     | 1          |
| propo 3, remi 5     | 1          |

### 6.4 Resumen de anotaciones por paciente

| Paciente     | #total | #estímulos | #medicación | #otros | Timestamps fuera de rango |
| ------------ | ------ | ---------- | ----------- | ------ | ------------------------- |
| 230393       | 18     | 4          | 0           | 14     | —                         |
| 397651       | 24     | 10         | 2           | 12     | —                         |
| 4214722      | 15     | 6          | 1           | 8      | —                         |
| 4234018 mal  | 0      | 0          | 0           | 0      | —                         |
| 4247699      | 24     | 9          | 3           | 12     | —                         |
| 4912692      | 16     | 5          | 2           | 9      | —                         |
| 5020549      | 17     | 8          | 1           | 8      | —                         |
| 5362391      | 7      | 4          | 1           | 2      | —                         |
| 5431482      | 17     | 11         | 2           | 4      | —                         |
| 5582912      | 19     | 9          | 2           | 8      | —                         |
| 5589679      | 15     | 7          | 3           | 5      | —                         |
| 5684023      | 7      | 4          | 1           | 2      | —                         |
| 70078466     | 0      | 0          | 0           | 0      | —                         |
| 70288016     | 20     | 6          | 0           | 14     | —                         |
| 70297385     | 8      | 3          | 1           | 4      | —                         |
| 70431992     | 14     | 6          | 0           | 8      | —                         |
| 70436283     | 11     | 8          | 0           | 3      | —                         |
| 70551555     | 16     | 6          | 2           | 8      | —                         |
| 70628874     | 13     | 2          | 1           | 10     | —                         |
| 70767707 mal | 2      | 1          | 0           | 1      | —                         |
| 720142       | 11     | 5          | 1           | 5      | —                         |

## 7. Reconciliación con análisis previo

El enunciado menciona "46.539 registros con eventos en 20 pacientes" de un
RESUMEN_RESULTADOS_ESTADISTICOS. **Este fichero no se ha encontrado en el repositorio.**
La reconciliación completa no es posible hasta que se aporte ese documento.

### Estimación de implicaciones del número 46.539

Si 46.539 representa el número de *ventanas de análisis* (no de lecturas NIBP):

| Supuesto | Implicación |
|----------|-------------|
| Ventanas de 30 s, paso 1 latido (BRS-like, ~1 s paso) | 46.539 ventanas / 20 pac = ~2.327 ventanas/pac → ~2.327 s = ~38.8 min de datos válidos/pac (razonable para cirugía 2-4 h) |
| Ventanas de 30 s, paso 30 s (sin solapamiento) | 46.539 × 30 s / 20 pac = ~23.270 s = ~388 min/pac → **muy alto para cirugía** |
| Lecturas NIBP directas | 46.539 / 20 = ~2.327 lecturas NIBP/pac → a 5 min/ciclo = ~194 h/pac → **imposible** |

**Hipótesis más probable**: los 46.539 registros son filas de un DataFrame con ventanas
deslizantes (paso ~1 latido ≈ 1 s) sobre señal PPG-derivada. Esto implica señal continua
reindexada, no lecturas NIBP brutas. Verificar cuando esté disponible el análisis previo.

### Inconsistencias a verificar
- Este script ha encontrado 21 carpetas de paciente. Si el análisis previo usó 20 pacientes
  exactamente, hay que identificar cuáles fueron incluidos/excluidos.
- Las carpetas marcadas "mal" sugieren exclusiones previas: **4234018 mal** (sin .vital) y
  **70767707 mal** (tiene .vital — verificar si fue excluido).

## 8. Factibilidad de features BeatLabile-like

| Feature BeatLabile                  | Disponible directamente                                                    | Surrogate propuesto                                                                                                               | Calidad surrogate                                                                      |
| ----------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| cv-PA-std                           | NO — requiere arteria invasiva                                             | Variabilidad de amplitud de pulso PPG (PPG-PAV); o variabilidad inter-ciclo de NIBP (escala temporal diferente: ~5 min vs latido) | Baja-Moderada — proxy indirecto, escala temporal incomparable                          |
| brs-min                             | NO — requiere PA sistólica latido-a-latido                                 | Cross-spectral coherence PPG-RR (PPG-BRS); o PTT variability vs RR si hay ECG + PPG                                               | Moderada — validado en literatura pero con limitaciones en perioperatorio              |
| HRV-SDNN                            | COMPUTABLE vía PRV (pulse rate variability) desde peaks PPG                | PRV-SDNN: SDNN de intervalos inter-pulso PPG                                                                                      | Moderada-Alta — PRV ≈ HRV en condiciones estables; sesgo en arritmias o artefactos PPG |
| HRV-RMSSD                           | COMPUTABLE vía PRV                                                         | PRV-RMSSD                                                                                                                         | Moderada-Alta — misma caveat que PRV-SDNN                                              |
| HRV-pNN50                           | COMPUTABLE vía PRV                                                         | PRV-pNN50                                                                                                                         | Moderada — sensible a calidad del detector de peaks PPG                                |
| RSA                                 | COMPUTABLE desde PRV si banda respiratoria identificable                   | Potencia HF (0.15-0.4 Hz) de PRV; o coherencia PPG-respiro si hay capnografía/resp                                                | Moderada — anestesia y bloqueantes reducen RSA; útil como marcador de efecto           |
| ARV de PA                           | COMPUTABLE macroscópicamente desde NIBP (escala 5 min, no latido-a-latido) | ARV-NIBP: suma |ΔMAP| entre lecturas consecutivas de NIBP / duración                                                              | Baja — ARV original es latido-a-latido; versión NIBP captura solo variabilidad lenta   |
| PPG-PAI (Perfusion Amplitude Index) | COMPUTABLE directamente — amplitud pico-valle PPG                          | Nativo PPG — no requiere surrogate                                                                                                | Alta para señal PPG de calidad — marcador de vasoconstricción/vasodilatación           |
| PRV-LF/HF ratio                     | COMPUTABLE desde PRV                                                       | Ratio potencia LF (0.04-0.15 Hz) / HF (0.15-0.4 Hz) de PRV                                                                        | Baja-Moderada — interpretación LF/HF muy debatida; anestesia confunde                  |

### Conclusión de factibilidad

Las features de BeatLabile original (cv-PA-std, brs-min, ARV) requieren arteria invasiva
y **no son computables en esta cohorte**. Los surrogates PPG/PRV son técnicamente posibles
pero introducen limitaciones conocidas:

1. **PRV ≠ HRV** durante anestesia, especialmente con cambios de volumen o vasopresores.
2. **La escala temporal del NIBP** (~5 min) es incomparable con la escala latido-a-latido
   de BeatLabile — las features basadas en NIBP miden variabilidad macroscópica, no
   la micro-variabilidad que caracteriza la firma autonómica.
3. El **PPG-PAI** (perfusion amplitude index) puede ser el mejor surrogate disponible:
   refleja tono vasomotor y puede detectar cambios autonómicos perioperatorios.
4. Sin ECG simultáneo, no es posible estimar PTT para BRS surrogate.

## 9. Sanity checks y Gantts por paciente

### 9.1 Verificación de sincronización temporal

Los tracks dentro de un mismo .vital comparten la línea de tiempo del VitalRecorder.
En principio están sincronizados. Los desfases observados en datos reales suelen deberse a:
- Reinicios de dispositivo durante la cirugía
- Segmentos con timestamps no monotónicos (se reportarían como gaps)
- El dispositivo VitalRecorder puede tener deriva del reloj < 1 s/h en grabaciones largas

Resultado de verificación: no se ha ejecutado análisis de monotonía de timestamps en
este script (requeriría iterar sobre todos los recs). Se recomienda verificar con
`sorted(rec.dt for rec in trk.recs)` vs `[rec.dt for rec in trk.recs]` antes del análisis.

### 9.2 Solapamiento anotaciones / gaps PPG

Verificación por paciente de si hay anotaciones en períodos donde el PPG está caído:

| Paciente     | #anotaciones | #gaps PPG >5s | #anotaciones en gap PPG |
| ------------ | ------------ | ------------- | ----------------------- |
| 230393       | 18           | 0             | 0                       |
| 397651       | 24           | 6             | 2                       |
| 4214722      | 15           | 0             | 0                       |
| 4234018 mal  | 0            | 0             | 0                       |
| 4247699      | 24           | 1             | 1                       |
| 4912692      | 16           | 0             | 0                       |
| 5020549      | 17           | 0             | 0                       |
| 5362391      | 7            | 1             | 0                       |
| 5431482      | 17           | 4             | 0                       |
| 5582912      | 19           | 0             | 0                       |
| 5589679      | 15           | 0             | 0                       |
| 5684023      | 7            | 5             | 3                       |
| 70078466     | 0            | 0             | 0                       |
| 70288016     | 20           | 0             | 0                       |
| 70297385     | 8            | 6             | 2                       |
| 70431992     | 14           | 0             | 0                       |
| 70436283     | 11           | 1             | 0                       |
| 70551555     | 16           | 4             | 2                       |
| 70628874     | 13           | 0             | 0                       |
| 70767707 mal | 2            | 0             | 0                       |
| 720142       | 11           | 0             | 0                       |

### 9.3 Diagramas Gantt por paciente

```
Leyenda: G=gap PPG  N=NIBP  A=anotación  -=datos continuos
```

**230393** (supra+axilar) — duración: 85.9 min | NIBP: 46 | Anotaciones: 18

```
  |A--N---N--NANNN--A-AANAAN--NAN-NA-AA-ANA-ANAANN-NA--N--N--A-|
   0m            21m            43m            64m            8
  Leyenda: G=gap PPG  N=NIBP  A=anotación  -=sin dato
```

**397651** (supra+axilar) — duración: 93.5 min | NIBP: 28 | Anotaciones: 24

```
  |NGAN-GA-GGN-AAAA-A-ANAN-AAAANAAN-ANN-A-A-NGAA--GGGGGGGGGANGG|
   0m            23m            47m            70m            9
  Leyenda: G=gap PPG  N=NIBP  A=anotación  -=sin dato
```

**4214722** (interescalenico) — duración: 78.4 min | NIBP: 34 | Anotaciones: 15

```
  |NN-ANN----N-N-AAN---N-ANA--AN--N---AA--N-AAN--AA--NA--N----A|
   0m            20m            39m            59m            7
  Leyenda: G=gap PPG  N=NIBP  A=anotación  -=sin dato
```

**4234018 mal** (interescalenico) — duración: 0.0 min | NIBP: 0 | Anotaciones: 0

```
  [duración desconocida]
```

**4247699** (interescalenico) — duración: 84.9 min | NIBP: 35 | Anotaciones: 24

```
  |A-AA---NA-AA-AN-AANA-NA--A--A---NA-N-A-A--N-A-N--AN--NAN----|
   0m            21m            42m            64m            8
  Leyenda: G=gap PPG  N=NIBP  A=anotación  -=sin dato
```

**4912692** (supra+axilar) — duración: 95.7 min | NIBP: 41 | Anotaciones: 16

```
  |N---NA-ANNA-A---N--AAANA-A-NAA-AN---A-N--N-AN-NN-ANN-NN--N--|
   0m            24m            48m            72m            9
  Leyenda: G=gap PPG  N=NIBP  A=anotación  -=sin dato
```

**5020549** (interescalenico) — duración: 83.4 min | NIBP: 34 | Anotaciones: 17

```
  |----N--ANN--A-AN--N-A-A--ANAAANA-NA-N---NAA-A--AA--N--N---A-|
   0m            21m            42m            63m            8
  Leyenda: G=gap PPG  N=NIBP  A=anotación  -=sin dato
```

**5362391** (interescalenico) — duración: 54.1 min | NIBP: 19 | Anotaciones: 7

```
  |N--N--A------GN----A----AN----NA-N--N----NN----A----AN------|
   0m            14m            27m            41m            5
  Leyenda: G=gap PPG  N=NIBP  A=anotación  -=sin dato
```

**5431482** (interescalenico) — duración: 92.5 min | NIBP: 36 | Anotaciones: 17

```
  |G-N--A-NNA--N-ANAAANA-NA-N--N-ANA-AN-AA-AN--NNA-NA-N--N--G--|
   0m            23m            46m            69m            9
  Leyenda: G=gap PPG  N=NIBP  A=anotación  -=sin dato
```

**5582912** (interescalenico) — duración: 85.7 min | NIBP: 41 | Anotaciones: 19

```
  |N--AN-ANAN-NAAN--AANAN-A-AA-N---NAANNANA--NN-AN--N---N--N--A|
   0m            21m            43m            64m            8
  Leyenda: G=gap PPG  N=NIBP  A=anotación  -=sin dato
```

**5589679** (interescalenico) — duración: 75.4 min | NIBP: 40 | Anotaciones: 15

```
  |---N---NA--N---N--NNA-AA-A-NNNAANANNNA-NA--A---AA--A---N----|
   0m            19m            38m            57m            7
  Leyenda: G=gap PPG  N=NIBP  A=anotación  -=sin dato
```

**5684023** (interescalenico) — duración: 76.7 min | NIBP: 19 | Anotaciones: 7

```
  |GGGGGGGGGGGGGGGGGGGAAN---A-ANN-GGA--GN--NN--NNA-N---NA-GGGN-|
   0m            19m            38m            58m            7
  Leyenda: G=gap PPG  N=NIBP  A=anotación  -=sin dato
```

**70078466** (supra+axilar) — duración: 0.0 min | NIBP: 0 | Anotaciones: 0

```
  [duración desconocida]
```

**70288016** (supra+axilar) — duración: 94.2 min | NIBP: 44 | Anotaciones: 20

```
  |-N--N-AN-NNNA-N-NAA-AA-A--NA-NAA-AAANA-NA-NA--A--A--N-ANN-N-|
   0m            24m            47m            71m            9
  Leyenda: G=gap PPG  N=NIBP  A=anotación  -=sin dato
```

**70297385** (interescalenico) — duración: 64.3 min | NIBP: 22 | Anotaciones: 8

```
  |---N-A---GGGAN-GANN---NAA-N----N--GGNA--NN-GGGA--NNANNGGGN--|
   0m            16m            32m            48m            6
  Leyenda: G=gap PPG  N=NIBP  A=anotación  -=sin dato
```

**70431992** (supra+axilar) — duración: 85.8 min | NIBP: 41 | Anotaciones: 14

```
  |N-N--N---N-NAN-AAA-N-A-A-NNA-AN-AN---N--N---N--A---N--NA--N-|
   0m            21m            43m            64m            8
  Leyenda: G=gap PPG  N=NIBP  A=anotación  -=sin dato
```

**70436283** (interescalenico) — duración: 65.5 min | NIBP: 24 | Anotaciones: 11

```
  |N---N--A-----AA--A----N-A-NNNA-NA--AN---A---AN---A----N-----|
   0m            16m            33m            49m            6
  Leyenda: G=gap PPG  N=NIBP  A=anotación  -=sin dato
```

**70551555** (supra+axilar) — duración: 97.4 min | NIBP: 44 | Anotaciones: 16

```
  |N----A-GGGGGANNA-ANN-AANA--NNAAN-N--AA-N--N-NNGGAA--NA-NA-N-|
   0m            24m            49m            73m            9
  Leyenda: G=gap PPG  N=NIBP  A=anotación  -=sin dato
```

**70628874** (supra+axilar) — duración: 70.3 min | NIBP: 44 | Anotaciones: 13

```
  |N--A-NN-NA-N-N--A-N--N-NAANAA-ANAN--N--A-N--N-AA-N-NNAN--N-N|
   0m            18m            35m            53m            7
  Leyenda: G=gap PPG  N=NIBP  A=anotación  -=sin dato
```

**70767707 mal** (interescalenico) — duración: 22.8 min | NIBP: 9 | Anotaciones: 2

```
  |NN------------NA-------------N------------N-N---A-------N---|
   0m            6m             11m            17m            2
  Leyenda: G=gap PPG  N=NIBP  A=anotación  -=sin dato
```

**720142** (interescalenico) — duración: 73.9 min | NIBP: 33 | Anotaciones: 11

```
  |-------N-NANN-ANNNNN---NA-NA-A-N-NNN--ANA--N---N-A-N-A-AA--N|
   0m            18m            37m            55m            7
  Leyenda: G=gap PPG  N=NIBP  A=anotación  -=sin dato
```

## 10. Recomendaciones operativas

### 10.1 Pacientes candidatos a exclusión

- **4234018 mal**: sin archivo .vital
- **70078466**: sin archivo .vital
- **5684023**: PPG válido 56.2% < 70%

### 10.2 Definiciones operativas a fijar antes del análisis

| Parámetro | Propuesta | Justificación |
|-----------|-----------|---------------|
| **Umbral hipotensión** | MAP < 65 mmHg (1 lectura) o MAP < 55 mmHg (2 lecturas consecutivas) | Sin arteria invasiva, no se puede exigir "3 min continuos"; proponer dos criterios y analizar sensibilidad |
| **Ventana pre-evento** | 30 min antes de primera lectura hipotensora | Consistente con BeatLabile; verificar que haya ≥10 lecturas NIBP en esa ventana |
| **Criterio de calidad PPG mínima** | ≥70% válido en la ventana de análisis | Umbral estándar; considerar elevar a 80% dado que SNR PPG perioperatorio es variable |
| **Definición "baja PA estable" (Q3)** | MAP ∈ [50-65] mmHg durante ≥2 lecturas consecutivas SIN tendencia a caída | Distinguir del "pre-hipotensión": la tendencia temporal (slope de NIBP) es clave |
| **Ventana de bloqueo activo** | Desde anotación de colocación + 20 min (onset típico) hasta fin de cirugía | Ajustar si hay anotación explícita de "inicio de bloqueo confirmado" |
| **Estimación PRV** | Peaks PPG con neurokit2, rechazo de peaks con Δt <200 ms o >2000 ms | Consistente con criterios de HRV estándar (RMSSD/SDNN) |

### 10.3 Riesgos identificados

1. **Potencia estadística (n=20)**: Con 12+8 pacientes, incluso un AUC de 0.80 con SD=0.15 tiene potencia <50% para detectarlo frente a 0.50. Se recomienda presentar intervalos de confianza amplios y no concluir sobre ausencia de efecto.
2. **Confounders no controlados**: BMI, HTA previa, medicación habitual, tipo de anestesia (propofol vs sevoflurano), duración de cirugía. Verificar disponibilidad en archivos .csv de cada paciente.
3. **Bloqueo interescalénico y PPG**: El bloqueo causa vasodilatación ipsilateral del brazo. Si el pulsioxímetro está en el brazo bloqueado, el PPG tendrá morfología alterada (mayor amplitud, menor variabilidad). **Verificar en qué brazo está el sensor SpO2**.
4. **Posición en silla de playa**: La posición semi-sentada puede causar hipotensión posicional y reducir la señal PPG. Considerar ajuste de valores de referencia.
5. **Desfibrilación/artefactos electroquirúrgicos**: La electrocirugía (fresa artroscópica) genera artefactos en todas las señales. Verificar correlación entre anotaciones de fresa y gaps en PPG.
6. **Sesgo de selección**: Si los "pacientes con bloqueo" tienen cirugías más largas (mayor complejidad) o diferente perfil de riesgo, el grupo no es comparable por duración.
7. **Transportabilidad**: BeatLabile demostró baja concordancia entre cohortes para hipotensión (ρ=0.188). Esta cohorte es aún más específica (hombro + bloqueo), lo que puede mejorar la homogeneidad intra-cohorte pero dificultar la generalización.

### 10.4 Próximos pasos recomendados

1. Confirmar asignación de grupos (13 bloqueo interescalénico / 8 bloqueo supra+axilar) — **cargada desde Excel**.
2. Verificar los archivos .csv de cada paciente (datos clínicos: BMI, HTA, ASA, etc.).
3. Localizar los datos de Medasense (NOL — nociception level) si aplican al subestudio.
4. Revisar manualmente las anotaciones de las carpetas con alertas de nombre ambiguo.
5. Definir si la carpeta "4234018 mal" (sin .vital) y "70767707 mal" (tiene .vital) son exclusiones previas o datos pendientes de cargar.

## Cuestiones abiertas

1. **Asignación de grupos**: Cargada desde `database general pacientes estudio hombro.xlsx` — 13 interescalénico / 8 supra+axilar. ✅ Resuelta.
2. **Archivos .vital con nombre de dispositivo** (patrón `scjrke68a_YYMMDD_HHMMSS.vital`): No contienen el ID de paciente en el nombre. La correspondencia se ha verificado manualmente por fecha y por `dtstart`. Marcados como "nombre ambiguo" solo como aviso informativo, no como error.
3. **Datos Medasense (NOL)**: Todas las carpetas tienen subcarpeta `Medasense_Data/`. ¿Se va a integrar el índice de nocicepción como covariable?
4. **Sensor PPG ipsilateral vs contralateral**: ¿El pulsioxímetro está en el brazo del bloqueo o en el contralateral? Crítico para interpretar cambios en PPG-PAI.
5. **Archivos .csv por paciente**: Contienen datos clínicos o exportaciones de VitalRecorder. ¿Hay información adicional de señal o solo metadatos?
6. **Sincronización Medasense-VitalRecorder**: Si se integran los datos NOL, verificar que los timestamps estén en la misma zona horaria y reference time.
7. **Paciente 4234018** (carpeta `4234018 mal`): Sin .vital — VitalRecorder se apagó antes de grabar. Confirmar si se excluye del análisis.
8. **Paciente 70767707** (carpeta `70767707 mal`): Grabación de solo 23 min — VitalRecorder se apagó. Insuficiente para features de ventana larga. Revisar si se incluye como parcial o se excluye.
9. **Paciente 70078466**: Sin .vital (archivo erróneo eliminado). Sin datos de señal recuperables. Excluir.
10. **Anotaciones en ficheros .csv**: Si las anotaciones no aparecen en el .vital (ver sección 6), pueden estar en los .csv exportados. Incluir parser de CSV en la próxima iteración.
