"""
Re-exporta el modelo de Bosques Aleatorios comprimido.

El .joblib que sale de Colab ocupa 61 MB porque joblib guarda los arrays de
los 300 árboles sin comprimir. Con compress=3 baja a unos 15-20 MB sin tocar
ni un parámetro del modelo: las predicciones son idénticas.

Útil para subir el proyecto a GitHub / Streamlit Cloud, donde un archivo de
61 MB dispara la advertencia de tamaño y hace lento cada despliegue.

Ejecutar una sola vez:  .venv/bin/python comprimir_modelo.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
ORIGEN = BASE / "Bosques aleatorios" / "modelo_rf.joblib"
DESTINO = BASE / "modelos" / "modelo_rf.joblib"

DESTINO.parent.mkdir(exist_ok=True)

print(f"Leyendo  {ORIGEN}  ({ORIGEN.stat().st_size / 1e6:.1f} MB)")
modelo = joblib.load(ORIGEN)

joblib.dump(modelo, DESTINO, compress=3)
print(f"Escrito  {DESTINO}  ({DESTINO.stat().st_size / 1e6:.1f} MB)")

# Verificación: las predicciones del modelo comprimido deben ser idénticas.
# Se usan registros reales del dataset para no disparar avisos de sklearn por
# pasar un array sin nombres de columna.
recargado = joblib.load(DESTINO)
datos = pd.read_csv(BASE / "AirQualityUCI_limpio.csv")
X = datos.sample(500, random_state=0)[list(modelo.feature_names_in_)]
iguales = np.allclose(modelo.predict(X), recargado.predict(X))
print("Predicciones idénticas tras comprimir:", iguales)

if not iguales:
    raise SystemExit("El modelo comprimido NO reproduce las predicciones. No lo uses.")

print(
    "\nListo. nucleo.py busca primero en 'modelos/', así que la app ya usará\n"
    "el comprimido. El original se puede dejar donde está (está en .gitignore)."
)
