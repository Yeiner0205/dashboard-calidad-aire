# ==========================================================
# CELDA PARA PEGAR AL FINAL DE CADA NOTEBOOK DE COLAB
# (o en un notebook nuevo donde cargues los 3 modelos)
# Exporta modelos + metadatos para el dashboard de Streamlit
# ==========================================================
import json, joblib, sklearn, numpy, pandas, os

os.makedirs("export", exist_ok=True)

# ----------------------------------------------------------
# 1) AJUSTA ESTOS NOMBRES a las variables de TUS notebooks
# ----------------------------------------------------------
modelos = {
    "random_forest": rf_model,      # Modelo 1
    "arbol_decision": dt_model,     # Modelo 2
    "svr_rbf": svr_model,           # Modelo 3
}

# El scaler que usaste para el SVR (si no escalaste, pon None)
scaler = scaler_svr        # p.ej. StandardScaler() ya ajustado
# Lista de columnas de entrada EN EL ORDEN EXACTO del entrenamiento
features = list(X_train.columns)
# Nombre de la variable objetivo
target = "CO(GT)"

# Métricas de test de cada modelo (rellena con los valores que ya calculaste)
metricas = {
    "random_forest":  {"R2": None, "RMSE": None, "MAE": None},
    "arbol_decision": {"R2": None, "RMSE": None, "MAE": None},
    "svr_rbf":        {"R2": None, "RMSE": None, "MAE": None},
}
# Marca True solo en los modelos que requieren datos escalados
usa_scaler = {"random_forest": False, "arbol_decision": False, "svr_rbf": True}

# ----------------------------------------------------------
# 2) EXPORTACIÓN (no tocar)
# ----------------------------------------------------------
for nombre, modelo in modelos.items():
    joblib.dump(modelo, f"export/{nombre}.joblib")
    print("guardado:", nombre, type(modelo).__name__)

if scaler is not None:
    joblib.dump(scaler, "export/scaler.joblib")
    print("guardado: scaler")

# Rango real de cada variable -> sirve para los sliders del dashboard
rangos = {
    c: {
        "min": float(X_train[c].min()),
        "max": float(X_train[c].max()),
        "media": float(X_train[c].mean()),
        "mediana": float(X_train[c].median()),
    }
    for c in features
}

meta = {
    "features": features,
    "target": target,
    "usa_scaler": usa_scaler,
    "metricas": metricas,
    "rangos": rangos,
    "versiones": {
        "sklearn": sklearn.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
    },
}
with open("export/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nVERSIONES EN COLAB:", meta["versiones"])
print("FEATURES:", features)

# Exporta también el dataset ya limpio (para las gráficas del dashboard)
df.to_csv("export/datos_limpios.csv", index=False)   # 'df' = tu DataFrame limpio

# ----------------------------------------------------------
# 3) DESCARGA UN ZIP CON TODO
# ----------------------------------------------------------
import shutil
from google.colab import files
shutil.make_archive("modelos_calidad_aire", "zip", "export")
files.download("modelos_calidad_aire.zip")
