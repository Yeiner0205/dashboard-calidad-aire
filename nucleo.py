"""
Núcleo del dashboard de predicción de CO(GT).

Centraliza la carga de datos y modelos, la construcción de las variables
derivadas y el cálculo de métricas. Todo replica exactamente lo que hacen
los cuadernos de Colab (mismo orden de features, mismo log natural, mismos
armónicos de hora y mismo train_test_split con random_state=42).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent

# --------------------------------------------------------------------------
# Definiciones tomadas de los cuadernos
# --------------------------------------------------------------------------
SENSORES = [
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
]
AMBIENTALES = ["T", "RH", "AH"]

# El orden importa: es el mismo con el que se entrenaron los tres modelos.
FEATURES = (
    SENSORES
    + ["log_" + c for c in SENSORES]
    + AMBIENTALES
    + ["hour_sin", "hour_cos", "hour_sin2", "hour_cos2"]
    + ["day_of_week"]
)
OBJETIVO = "CO(GT)"

DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

# Cada sensor PT08 se nombra por el gas al que apunta, no por su material.
# La documentación del dataset precisa que el gas objetivo es "nominal": son
# sensores de óxido metálico con sensibilidad cruzada, responden a varios gases
# y su señal se desplaza con la temperatura y la humedad. Por eso ningún sensor
# sirve por sí solo y el modelo necesita los cinco más T, RH y AH.
ETIQUETAS_SENSOR = {
    "PT08.S1(CO)": "Sensor S1 · apunta a monóxido de carbono (material: óxido de estaño)",
    "PT08.S2(NMHC)": "Sensor S2 · apunta a hidrocarburos no metánicos (material: titania)",
    "PT08.S3(NOx)": "Sensor S3 · apunta a óxidos de nitrógeno (material: óxido de tungsteno)",
    "PT08.S4(NO2)": "Sensor S4 · apunta a dióxido de nitrógeno (material: óxido de tungsteno)",
    "PT08.S5(O3)": "Sensor S5 · apunta a ozono (material: óxido de indio)",
    "T": "Temperatura (°C)",
    "RH": "Humedad relativa (%)",
    "AH": "Humedad absoluta",
}

# Rutas de los modelos. Se busca primero en 'modelos/' (versiones comprimidas
# para el despliegue) y si no está, en las carpetas originales de cada cuaderno.
RUTAS_MODELOS = {
    "Bosques Aleatorios": ["modelos/modelo_rf.joblib", "Bosques aleatorios/modelo_rf.joblib"],
    "Árbol de Decisión": ["modelos/modelo_dt.joblib", "Arboles de decision/modelo_dt.joblib"],
    "SVR": ["modelos/modelo_svr.joblib", "Support vector/modelo_svr.joblib"],
}
RUTAS_DATOS = ["AirQualityUCI_limpio.csv", "datos/AirQualityUCI_limpio.csv"]

BASE_LINEAL = "Regresión Lineal (línea base)"

COLORES = {
    "Bosques Aleatorios": "#2E7D32",
    "Árbol de Decisión": "#F9A825",
    "SVR": "#1565C0",
    BASE_LINEAL: "#9E9E9E",
}

# Hiperparámetros declarados en cada cuaderno, para mostrarlos en la interfaz.
CONFIGURACION = {
    "Bosques Aleatorios": "n_estimators=300, max_depth=15, min_samples_split=4, min_samples_leaf=2",
    "Árbol de Decisión": "criterion=squared_error, max_depth=10, max_leaf_nodes=1000, min_samples_leaf=4",
    "SVR": "Pipeline: StandardScaler + SVR(kernel=rbf, C=10.0, epsilon=0.1, gamma=scale)",
    BASE_LINEAL: "Mínimos cuadrados ordinarios sobre las 18 variables",
}


def _resolver(candidatas):
    """Devuelve la primera ruta que exista, o lanza un error explicativo."""
    for rel in candidatas:
        ruta = BASE / rel
        if ruta.exists():
            return ruta
    raise FileNotFoundError(
        "No se encontró ninguno de estos archivos:\n  - "
        + "\n  - ".join(candidatas)
    )


# --------------------------------------------------------------------------
# Carga de datos
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando el dataset…")
def cargar_datos() -> pd.DataFrame:
    """Lee el CSV limpio y recupera la hora del día desde los armónicos."""
    df = pd.read_csv(_resolver(RUTAS_DATOS))

    faltantes = [c for c in FEATURES + [OBJETIVO] if c not in df.columns]
    if faltantes:
        raise ValueError(f"El CSV no tiene estas columnas: {faltantes}")

    # hour_sin = sin(2πh/24) y hour_cos = cos(2πh/24) determinan la hora de
    # forma única, así que se invierte con atan2 para poder etiquetar y agrupar.
    angulo = np.arctan2(df["hour_sin"], df["hour_cos"])
    df["hora"] = np.rint(angulo * 24 / (2 * np.pi)).astype(int) % 24
    df["dia_nombre"] = df["day_of_week"].astype(int).map(dict(enumerate(DIAS)))
    return df


@st.cache_data
def rangos(df_hash: int = 0) -> pd.DataFrame:
    """Estadísticos por variable, usados para los límites de los controles."""
    df = cargar_datos()
    return df[SENSORES + AMBIENTALES + [OBJETIVO]].describe().T


# --------------------------------------------------------------------------
# Carga de modelos
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando los modelos entrenados…")
def cargar_modelos() -> dict:
    """
    Carga los tres modelos exportados y entrena la línea base lineal.

    La línea base no se exportó desde Colab, pero al ser determinista y usar
    el mismo split (random_state=42) reproduce las métricas del informe.
    """
    modelos = {nombre: joblib.load(_resolver(rutas)) for nombre, rutas in RUTAS_MODELOS.items()}

    x_train, _, y_train, _ = particion()
    modelos[BASE_LINEAL] = LinearRegression().fit(x_train, y_train)
    return modelos


@st.cache_data
def particion():
    """Reproduce el split 80/20 de los cuadernos."""
    df = cargar_datos()
    return train_test_split(
        df[FEATURES], df[OBJETIVO], test_size=0.2, random_state=42
    )


@st.cache_data
def indices_prueba() -> set:
    """
    Índices que cayeron en el conjunto de prueba.

    Sirve para avisar en la interfaz cuando un registro ya fue visto por los
    modelos durante el entrenamiento: en esos la predicción es optimista y no
    mide la capacidad real de generalizar.
    """
    _, x_test, _, _ = particion()
    return set(x_test.index)


# --------------------------------------------------------------------------
# Construcción de las variables de entrada
# --------------------------------------------------------------------------
def construir_entrada(valores: dict, hora: int, dia_semana: int) -> pd.DataFrame:
    """
    Arma una fila con las 18 features a partir de los valores crudos.

    'valores' trae los 5 sensores y T/RH/AH. Los logaritmos y los armónicos
    de hora se derivan aquí con las mismas fórmulas de los cuadernos.
    """
    fila = {c: float(valores[c]) for c in SENSORES + AMBIENTALES}

    for c in SENSORES:
        # Los sensores PT08 responden de forma aproximadamente logarítmica.
        fila["log_" + c] = float(np.log(fila[c]))

    fila["hour_sin"] = float(np.sin(2 * np.pi * hora / 24))
    fila["hour_cos"] = float(np.cos(2 * np.pi * hora / 24))
    fila["hour_sin2"] = float(np.sin(4 * np.pi * hora / 24))
    fila["hour_cos2"] = float(np.cos(4 * np.pi * hora / 24))
    fila["day_of_week"] = int(dia_semana)

    return pd.DataFrame([fila])[FEATURES]


def predecir(modelos: dict, X: pd.DataFrame, seleccionados=None) -> dict:
    """Predice con los modelos indicados y devuelve {nombre: array}."""
    nombres = seleccionados if seleccionados else list(modelos)
    return {n: modelos[n].predict(X[FEATURES]) for n in nombres}


def barrido_horario(modelos: dict, valores: dict, dia_semana: int, seleccionados) -> pd.DataFrame:
    """
    Predice las 24 horas del día manteniendo fijas las lecturas de sensores.

    Sirve para ver el patrón diario que cada modelo aprendió de los armónicos.
    """
    X = pd.concat(
        [construir_entrada(valores, h, dia_semana) for h in range(24)], ignore_index=True
    )
    salida = pd.DataFrame({"Hora": range(24)})
    for nombre, pred in predecir(modelos, X, seleccionados).items():
        salida[nombre] = pred
    return salida


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Evaluando los modelos sobre el conjunto de prueba…")
def metricas() -> pd.DataFrame:
    """Recalcula las métricas de prueba y entrenamiento de cada modelo."""
    modelos = cargar_modelos()
    x_train, x_test, y_train, y_test = particion()

    filas = []
    for nombre, modelo in modelos.items():
        pred = modelo.predict(x_test)
        filas.append(
            {
                "Modelo": nombre,
                "MAE": mean_absolute_error(y_test, pred),
                "MSE": mean_squared_error(y_test, pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                "R² entrenamiento": r2_score(y_train, modelo.predict(x_train)),
                "R² prueba": r2_score(y_test, pred),
                "MAPE (%)": mean_absolute_percentage_error(y_test, pred) * 100,
            }
        )

    tabla = pd.DataFrame(filas)
    tabla["Brecha ΔR²"] = tabla["R² entrenamiento"] - tabla["R² prueba"]
    return tabla


@st.cache_data(show_spinner="Calculando predicciones del conjunto de prueba…")
def predicciones_prueba() -> pd.DataFrame:
    """Tabla larga con real vs predicho en test, para los gráficos de diagnóstico."""
    modelos = cargar_modelos()
    _, x_test, _, y_test = particion()

    trozos = []
    for nombre, modelo in modelos.items():
        pred = modelo.predict(x_test)
        trozos.append(
            pd.DataFrame(
                {
                    "Modelo": nombre,
                    "Real": y_test.to_numpy(),
                    "Predicho": pred,
                    "Residuo": y_test.to_numpy() - pred,
                }
            )
        )
    return pd.concat(trozos, ignore_index=True)


@st.cache_data
def importancias() -> pd.DataFrame:
    """
    Importancia de variables de los modelos basados en árboles.

    El SVR no expone importancias (habría que calcular
    permutation_importance, demasiado lento para una app interactiva).
    """
    modelos = cargar_modelos()
    trozos = []
    for nombre in ["Bosques Aleatorios", "Árbol de Decisión"]:
        modelo = modelos[nombre]
        trozos.append(
            pd.DataFrame(
                {
                    "Modelo": nombre,
                    "Variable": FEATURES,
                    "Importancia": modelo.feature_importances_,
                }
            )
        )
    return pd.concat(trozos, ignore_index=True)


# --------------------------------------------------------------------------
# Utilidades para la predicción por lotes
# --------------------------------------------------------------------------
def preparar_lote(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza un CSV subido por el usuario al formato de las 18 features.

    Acepta dos formatos:
      a) ya trae las 18 columnas derivadas (como el CSV limpio del proyecto);
      b) trae los 5 sensores + T/RH/AH + 'hora' y 'day_of_week' en crudo.
    """
    if all(c in df.columns for c in FEATURES):
        return df[FEATURES].copy()

    crudas = SENSORES + AMBIENTALES + ["hora", "day_of_week"]
    faltantes = [c for c in crudas if c not in df.columns]
    if faltantes:
        raise ValueError(
            "El CSV debe traer las 18 variables derivadas, o bien estas columnas "
            f"en crudo: {crudas}. Faltan: {faltantes}"
        )

    out = df[SENSORES + AMBIENTALES].astype(float).copy()
    for c in SENSORES:
        out["log_" + c] = np.log(out[c])

    h = df["hora"].astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * h / 24)
    out["hour_cos"] = np.cos(2 * np.pi * h / 24)
    out["hour_sin2"] = np.sin(4 * np.pi * h / 24)
    out["hour_cos2"] = np.cos(4 * np.pi * h / 24)
    out["day_of_week"] = df["day_of_week"].astype(int)

    return out[FEATURES]


# --------------------------------------------------------------------------
# Interpretación del valor predicho
# --------------------------------------------------------------------------
# Los cortes NO son arbitrarios: se apoyan en las guías de calidad del aire
# de la OMS (2021) y en la Directiva 2008/50/CE de la Unión Europea.
#
#   4 mg/m³  → guía de la OMS para la media de 24 horas.
#  10 mg/m³  → límite de la OMS y de la UE para la media de 8 horas.
#
# Advertencia importante: esas guías se definen sobre promedios de 24 u 8
# horas, mientras que CO(GT) es una medición horaria. La comparación es
# orientativa, no un dictamen legal de cumplimiento normativo.
NIVELES = [
    {
        "limite": 2.0,
        "nivel": "Aire limpio",
        "veredicto": "No hay contaminación",
        "color": "#2E7D32",
        "icono": "🟢",
        "mensaje": (
            "La concentración de monóxido de carbono es baja y está muy por debajo "
            "de todas las guías de calidad del aire. El aire se puede considerar limpio."
        ),
        "referencia": "Menos de la mitad de la guía de la OMS para 24 horas (4 mg/m³).",
    },
    {
        "limite": 4.0,
        "nivel": "Contaminación leve",
        "veredicto": "Hay algo de contaminación, pero dentro de lo aceptable",
        "color": "#F9A825",
        "icono": "🟡",
        "mensaje": (
            "Hay contaminación perceptible, la típica de una zona urbana con tráfico. "
            "Aun así el valor sigue por debajo del límite recomendado y no supone un "
            "riesgo para una persona sana."
        ),
        "referencia": "Por debajo de la guía de la OMS para 24 horas (4 mg/m³).",
    },
    {
        "limite": 10.0,
        "nivel": "Contaminación alta",
        "veredicto": "Sí hay contaminación",
        "color": "#EF6C00",
        "icono": "🟠",
        "mensaje": (
            "El aire está contaminado. Si esta concentración se mantiene durante horas, "
            "puede afectar a personas con problemas cardíacos o respiratorios, a mayores "
            "y a mujeres embarazadas. Conviene evitar el ejercicio intenso al aire libre."
        ),
        "referencia": "Supera la guía de la OMS para 24 horas (4 mg/m³).",
    },
    {
        "limite": float("inf"),
        "nivel": "Contaminación muy alta",
        "veredicto": "Sí hay contaminación, en nivel grave",
        "color": "#C62828",
        "icono": "🔴",
        "mensaje": (
            "Episodio grave de contaminación. A esta concentración el monóxido de carbono "
            "reduce la capacidad de la sangre para transportar oxígeno. Se desaconseja la "
            "actividad física al aire libre para cualquier persona."
        ),
        "referencia": "Alcanza el límite de la OMS y de la UE para 8 horas (10 mg/m³).",
    },
]


def interpretar(valor: float) -> dict:
    """Traduce una concentración de CO en mg/m³ a un veredicto entendible."""
    for nivel in NIVELES:
        if valor < nivel["limite"]:
            return nivel
    return NIVELES[-1]


def clasificar(valor: float) -> tuple:
    """Compatibilidad: devuelve solo (nivel, color)."""
    n = interpretar(valor)
    return n["nivel"], n["color"]
