# Dashboard de predicción de CO(GT) — Calidad del aire

Dashboard en Streamlit que monta los tres modelos de regresión entrenados sobre
el dataset **AirQualityUCI** para predecir la concentración de monóxido de
carbono `CO(GT)` en mg/m³.

Integrantes: Yeiner Pérez Vitola y Jhojan Marín Acosta.

## Modelos

| Modelo | Configuración | R² prueba |
|---|---|---|
| Bosques Aleatorios | `n_estimators=300, max_depth=15, min_samples_split=4, min_samples_leaf=2` | 0,8776 |
| Árbol de Decisión | `max_depth=10, max_leaf_nodes=1000, min_samples_leaf=4` | 0,8482 |
| SVR | `Pipeline(StandardScaler + SVR(C=10.0, epsilon=0.1, gamma=scale))` | 0,8794 |
| Regresión Lineal (línea base) | Mínimos cuadrados ordinarios | 0,8435 |

La línea base no se exportó desde Colab: la app la reentrena al arrancar sobre
el mismo split, lo que toma milisegundos y reproduce las métricas del informe.

## Instalación

```bash
sudo apt install -y python3-pip python3-venv
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Ejecución

```bash
.venv/bin/streamlit run app.py
```

Se abre en `http://localhost:8501`.

## Estructura

```
app.py                      Interfaz: las 5 pestañas del dashboard
nucleo.py                   Carga de datos y modelos, features derivadas, métricas
comprimir_modelo.py         Re-exporta el Random Forest comprimido (61 MB → ~18 MB)
requirements.txt            Dependencias (scikit-learn 1.6.1, igual que en Colab)
AirQualityUCI_limpio.csv    7.674 registros, 18 features + CO(GT)
Bosques aleatorios/         modelo_rf.joblib, features.joblib, cuaderno
Arboles de decision/        modelo_dt.joblib, cuaderno
Support vector/             modelo_svr.joblib, cuaderno
modelos/                    (opcional) versiones comprimidas para el despliegue
```

`nucleo.py` busca cada modelo primero en `modelos/` y, si no está, en la
carpeta original de su cuaderno.

## Pestañas

1. **Predicción manual** — sliders para los 5 sensores PT08 y las condiciones
   ambientales, más hora y día. Muestra la predicción de cada modelo, el nivel
   de riesgo y un barrido de las 24 horas con las mismas lecturas, que deja ver
   el ciclo diario que cada modelo aprendió.
2. **Registro del dataset** — selecciona un registro real y compara la
   predicción de cada modelo contra el valor medido.
3. **Predicción por lotes** — sube un CSV, predice todas las filas y descarga
   los resultados. Acepta el formato con las 18 features ya calculadas o las
   columnas en crudo (`PT08.S*`, `T`, `RH`, `AH`, `hora`, `day_of_week`).
4. **Comparación de modelos** — métricas recalculadas en vivo, real vs
   predicho, distribución de residuos e importancia de variables.
5. **Exploración de datos** — patrón horario y semanal, distribuciones,
   correlaciones.

## Interpretación de los valores

El dashboard traduce la concentración predicha a un veredicto en lenguaje
llano. Los umbrales están tomados de las guías de calidad del aire de la **OMS
(2021)** y de la **Directiva 2008/50/CE** de la Unión Europea, no son
arbitrarios:

| CO(GT) | Nivel | Veredicto | Referencia |
|---|---|---|---|
| < 2 mg/m³ | 🟢 Aire limpio | No hay contaminación | Menos de la mitad de la guía OMS de 24 h |
| 2 – 4 mg/m³ | 🟡 Contaminación leve | Hay algo, dentro de lo aceptable | Por debajo de la guía OMS de 24 h (4 mg/m³) |
| 4 – 10 mg/m³ | 🟠 Contaminación alta | Sí hay contaminación | Supera la guía OMS de 24 h |
| ≥ 10 mg/m³ | 🔴 Contaminación muy alta | Sí, en nivel grave | Alcanza el límite OMS/UE de 8 h (10 mg/m³) |

En el dataset, el 53,8 % de los registros cae en el primer nivel, el 34,7 % en
el segundo, el 11,5 % en el tercero y solo 5 registros (0,1 %) en el cuarto.

**Salvedad importante:** las guías de la OMS y de la UE se definen sobre
promedios de 24 u 8 horas, mientras que `CO(GT)` es una medición horaria. La
comparación es orientativa para dar contexto al usuario, no un dictamen de
cumplimiento normativo.

El veredicto se calcula sobre el promedio de los modelos seleccionados. Cuando
los modelos caen en niveles distintos (ocurre en el 11,4 % de los registros,
casi siempre porque el valor está pegado a un umbral), la app lo avisa en vez
de dar un veredicto único.

## Variables del modelo

Las 18 variables predictoras, en el orden exacto con el que se entrenaron:

```
PT08.S1(CO)  PT08.S2(NMHC)  PT08.S3(NOx)  PT08.S4(NO2)  PT08.S5(O3)
log_PT08.S1(CO) … log_PT08.S5(O3)          (logaritmo natural)
T  RH  AH
hour_sin  hour_cos  hour_sin2  hour_cos2   (hora cíclica, dos armónicos)
day_of_week                                (0 = lunes)
```

El usuario solo ingresa 9 valores más hora y día; las otras 9 variables se
derivan en `nucleo.construir_entrada()` con las mismas fórmulas de los
cuadernos:

- `log_X = np.log(X)` — los sensores de óxido metálico responden de forma
  aproximadamente logarítmica a la concentración.
- `hour_sin = sin(2πh/24)`, `hour_cos = cos(2πh/24)` y el segundo armónico
  `hour_sin2 = sin(4πh/24)`, `hour_cos2 = cos(4πh/24)` — la hora es cíclica
  (23:00 y 00:00 son contiguas) y el patrón diario tiene dos picos de tráfico,
  a las 8-9 AM y a las 7-8 PM.

El objetivo `CO(GT)` **no** está transformado, así que las predicciones salen
directamente en mg/m³.

Las métricas que muestra la app se recalculan reproduciendo el split de los
cuadernos (`train_test_split(test_size=0.2, random_state=42)` sobre el CSV, que
conserva el orden original de los registros), por lo que coinciden con las del
informe.

## Despliegue en Streamlit Cloud

1. Reducir el Random Forest, que de otro modo dispara la advertencia de tamaño
   de GitHub y hace lento cada despliegue:

   ```bash
   .venv/bin/python comprimir_modelo.py
   ```

2. Subir el repo. `.gitignore` ya excluye el `.joblib` original de 61 MB, así
   que se publica la versión comprimida de `modelos/`.

3. En [share.streamlit.io](https://share.streamlit.io) apuntar al repo con
   `app.py` como archivo principal. `requirements.txt` fija scikit-learn 1.6.1,
   la misma versión de Colab, para que los `.joblib` carguen sin advertencias
   de incompatibilidad.
