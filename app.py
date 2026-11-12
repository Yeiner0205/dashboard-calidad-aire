"""
Dashboard de predicción de CO(GT) — Calidad del aire (dataset AirQualityUCI).

Monta los tres modelos de regresión entrenados en Colab (Bosques Aleatorios,
Árbol de Decisión y SVR) más una línea base de Regresión Lineal,
y permite predecir con entrada manual, con registros reales del dataset o
cargando un CSV para predicción por lotes.

Ejecutar con:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import nucleo as nk

st.set_page_config(
    page_title="Predicción de CO — Calidad del aire",
    page_icon="🌫️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Carga inicial
# --------------------------------------------------------------------------
try:
    df = nk.cargar_datos()
    modelos = nk.cargar_modelos()
    tabla_metricas = nk.metricas()
except FileNotFoundError as e:
    st.error(f"Falta un archivo del proyecto.\n\n{e}")
    st.stop()

stats = nk.rangos()
medianas = {c: float(stats.loc[c, "50%"]) for c in nk.SENSORES + nk.AMBIENTALES}

st.title("🌫️ Predicción de monóxido de carbono CO(GT)")
st.caption(
    "Dataset AirQualityUCI · 7.674 registros reales · 18 variables predictoras · "
    "Tres modelos de regresión comparados contra una línea base lineal"
)

with st.expander("❓ ¿Qué estamos prediciendo exactamente y qué significa mg/m³?"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
**El monóxido de carbono (CO)** es un gas incoloro e inodoro que se produce
cuando algo se quema sin oxígeno suficiente. En una ciudad, la fuente principal
son los tubos de escape de los vehículos. No se ve ni se huele, y por eso hace
falta medirlo.

**mg/m³** significa *miligramos de CO por metro cúbico de aire*: cuánta masa de
gas hay en un volumen determinado. Un valor de 2 mg/m³ quiere decir que en cada
metro cúbico de aire hay 2 milésimas de gramo de monóxido de carbono.
            """
        )
    with col_b:
        st.markdown(
            """
**`CO(GT)` es la variable que predicen los modelos.** Las siglas GT vienen de
*ground truth*: es la concentración real medida por un analizador de referencia
certificado, el instrumento caro y preciso de la estación.

**¿Y para qué predecirla, si ya se mide?** Porque la estación también lleva
cinco sensores químicos baratos (`PT08.S1` a `PT08.S5`). Esos son los que se
pueden desplegar por toda una ciudad, pero no dan la concentración directamente:
dan una señal eléctrica que se desvía con la temperatura y la humedad.

El modelo aprende a traducir las lecturas de los sensores baratos al valor que
habría dado el analizador caro. Es decir, **calibra sensores económicos para que
funcionen como un instrumento de referencia**.
            """
        )
    st.info(
        "En resumen: a partir de 5 sensores baratos, la temperatura, la humedad y el "
        "momento del día, el modelo estima cuánto monóxido de carbono hay realmente "
        "en el aire — y abajo te dice si esa cantidad implica contaminación o no."
    )

# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuración")

    seleccionados = st.multiselect(
        "Modelos a usar",
        options=list(modelos),
        default=list(nk.RUTAS_MODELOS),
        help="Se aplican a todas las pestañas de predicción.",
    )
    if not seleccionados:
        st.warning("Selecciona al menos un modelo.")
        st.stop()

    st.divider()
    mejor = tabla_metricas.sort_values("R² prueba", ascending=False).iloc[0]
    st.metric("Mejor modelo (R² prueba)", mejor["Modelo"], f"{mejor['R² prueba']:.4f}")

    st.divider()
    with st.expander("Variables del modelo"):
        st.write(
            "El formulario pide **9 valores** (5 sensores + temperatura, humedad "
            "relativa y absoluta) más **hora y día de la semana**. Las otras 9 "
            "variables se derivan automáticamente:"
        )
        st.markdown(
            "- `log_*` de cada sensor: logaritmo natural, porque los sensores de "
            "óxido metálico responden de forma aproximadamente logarítmica.\n"
            "- `hour_sin`, `hour_cos`, `hour_sin2`, `hour_cos2`: la hora como "
            "variable cíclica con dos armónicos, por los dos picos de tráfico "
            "(8-9 AM y 7-8 PM)."
        )

    st.caption("Integrantes: Yeiner Pérez Vitola y Jhojan Marín Acosta")

tab_manual, tab_registro, tab_lote, tab_comparar, tab_datos = st.tabs(
    [
        "🎛️ Predicción manual",
        "📄 Registro del dataset",
        "📦 Predicción por lotes",
        "📊 Comparación de modelos",
        "🔎 Exploración de datos",
    ]
)


def mostrar_predicciones(preds: dict, real: float | None = None):
    """Pinta una tarjeta por modelo con su predicción y el nivel de riesgo."""
    columnas = st.columns(len(preds))
    for col, (nombre, valor) in zip(columnas, preds.items()):
        v = float(np.ravel(valor)[0])
        nivel, color = nk.clasificar(v)
        delta = f"{v - real:+.2f} vs real" if real is not None else nivel
        with col:
            st.metric(nombre, f"{v:.2f} mg/m³", delta, delta_color="off")
            st.markdown(
                f"<div style='height:6px;background:{color};border-radius:3px'></div>",
                unsafe_allow_html=True,
            )


def mostrar_veredicto(valores: list, etiqueta: str = "valor predicho"):
    """
    Banner que traduce la concentración a un veredicto en lenguaje llano.

    Se calcula sobre el promedio de los modelos seleccionados y avisa cuando
    no todos coinciden en el mismo nivel de contaminación.
    """
    promedio = float(np.mean(valores))
    n = nk.interpretar(promedio)

    st.markdown(
        f"""
        <div style="border-left:10px solid {n['color']};background:{n['color']}14;
                    padding:18px 22px;border-radius:8px;margin:10px 0 6px 0">
          <div style="font-size:1.45rem;font-weight:700;color:{n['color']};line-height:1.3">
            {n['icono']} {n['veredicto']}
          </div>
          <div style="font-size:1.02rem;margin-top:8px;color:#1B1B1B">{n['mensaje']}</div>
          <div style="font-size:0.88rem;margin-top:10px;color:#555">
            <b>{n['nivel']}</b> · {promedio:.2f} mg/m³ de media entre los modelos ({etiqueta})
            · {n['referencia']}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    niveles = {nk.interpretar(v)["nivel"] for v in valores}
    if len(niveles) > 1:
        st.warning(
            "Los modelos no coinciden en el nivel: "
            + ", ".join(sorted(niveles))
            + ". El valor está cerca de un umbral, así que conviene mirar cada predicción."
        )


def leyenda_niveles():
    """Tabla de referencia con los cuatro niveles y su origen normativo."""
    st.markdown("**Cómo se interpretan los valores**")
    st.dataframe(
        pd.DataFrame(
            {
                "Nivel": [f"{n['icono']} {n['nivel']}" for n in nk.NIVELES],
                "CO(GT)": ["menos de 2", "de 2 a 4", "de 4 a 10", "10 o más"],
                "¿Hay contaminación?": [n["veredicto"] for n in nk.NIVELES],
                "Referencia": [n["referencia"] for n in nk.NIVELES],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Umbrales tomados de las guías de calidad del aire de la OMS (2021) y de la "
        "Directiva 2008/50/CE de la Unión Europea. Ojo: esas guías se definen sobre "
        "promedios de 24 u 8 horas, mientras que CO(GT) es una medición horaria, así "
        "que la comparación es orientativa y no un dictamen de cumplimiento normativo."
    )


# ==========================================================================
# PESTAÑA 1 — Entrada manual
# ==========================================================================
with tab_manual:
    st.subheader("Ingresa las lecturas de los sensores")

    # Los controles se inicializan en session_state y luego se crean solo con
    # 'key'. Así el botón de restablecer puede reescribir sus valores sin que
    # Streamlit reclame por fijar un default y el estado a la vez.
    for c in nk.SENSORES + nk.AMBIENTALES:
        st.session_state.setdefault(f"in_{c}", medianas[c])
    st.session_state.setdefault("in_hora", 8)
    st.session_state.setdefault("in_dia", nk.DIAS[0])

    if st.button("↺ Restablecer a los valores medianos del dataset"):
        for c in nk.SENSORES + nk.AMBIENTALES:
            st.session_state[f"in_{c}"] = medianas[c]
        st.session_state["in_hora"] = 8
        st.session_state["in_dia"] = nk.DIAS[0]

    col_izq, col_der = st.columns(2)

    valores = {}
    with col_izq:
        st.markdown("**Sensores químicos (respuesta en unidades del sensor)**")
        for c in nk.SENSORES:
            valores[c] = st.slider(
                nk.ETIQUETAS_SENSOR[c],
                min_value=float(stats.loc[c, "min"]),
                max_value=float(stats.loc[c, "max"]),
                step=1.0,
                key=f"in_{c}",
            )

    with col_der:
        st.markdown("**Condiciones ambientales**")
        for c in nk.AMBIENTALES:
            valores[c] = st.slider(
                nk.ETIQUETAS_SENSOR[c],
                min_value=float(stats.loc[c, "min"]),
                max_value=float(stats.loc[c, "max"]),
                step=0.001 if c == "AH" else 0.1,
                key=f"in_{c}",
            )

        st.markdown("**Momento de la medición**")
        hora = st.slider("Hora del día", 0, 23, key="in_hora")
        dia_nombre = st.selectbox("Día de la semana", nk.DIAS, key="in_dia")
        dia = nk.DIAS.index(dia_nombre)

    X = nk.construir_entrada(valores, hora, dia)
    preds = nk.predecir(modelos, X, seleccionados)

    st.divider()
    st.subheader(f"Predicción para {dia_nombre.lower()} a las {hora:02d}:00")
    mostrar_predicciones(preds)

    valores_pred = [float(np.ravel(v)[0]) for v in preds.values()]
    if len(valores_pred) > 1:
        st.caption(
            f"Dispersión entre modelos (máx − mín): "
            f"**{max(valores_pred) - min(valores_pred):.2f} mg/m³**"
        )

    mostrar_veredicto(valores_pred)
    with st.expander("Ver la escala completa de niveles"):
        leyenda_niveles()

    st.divider()
    st.subheader("Patrón diario con estas mismas lecturas")
    st.caption(
        "Se mantienen fijos los sensores y las condiciones ambientales, y se varía "
        "solo la hora. Muestra el ciclo diario que cada modelo aprendió."
    )
    barrido = nk.barrido_horario(modelos, valores, dia, seleccionados)
    fig = px.line(
        barrido,
        x="Hora",
        y=seleccionados,
        markers=True,
        labels={"value": "CO(GT) predicho (mg/m³)", "variable": "Modelo"},
        color_discrete_map=nk.COLORES,
    )
    fig.add_vline(x=hora, line_dash="dash", line_color="gray", annotation_text="hora elegida")
    fig.update_layout(height=380, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver las 18 variables que reciben los modelos"):
        st.dataframe(X.T.rename(columns={0: "Valor"}), use_container_width=True)

# ==========================================================================
# PESTAÑA 2 — Registro real del dataset
# ==========================================================================
with tab_registro:
    st.subheader("Elige un registro medido y compara contra el valor real")
    st.markdown(
        "Cada **registro** es una hora real medida en la estación: las lecturas de los "
        "cinco sensores, la temperatura y la humedad de ese momento, y el CO(GT) que "
        "marcó el analizador de referencia. Aquí se le dan al modelo solo los sensores "
        "y se compara su predicción contra lo que de verdad se midió."
    )

    st.session_state.setdefault("idx_registro", 0)
    idx_test = nk.indices_prueba()

    solo_prueba = st.checkbox(
        "Mostrar solo registros del conjunto de prueba",
        value=False,
        help=(
            "El 80 % de los registros se usó para entrenar: en esos el modelo ya vio "
            "la respuesta, así que acierta más de lo que acertaría con datos nuevos. "
            "Los de prueba son los que miden la capacidad real de generalizar."
        ),
    )

    col_a, col_b = st.columns([3, 1])
    # El botón se declara antes del slider para poder reescribir su estado:
    # Streamlit no permite modificar la clave de un widget ya instanciado.
    with col_b:
        if st.button("🎲 Registro aleatorio"):
            universo = sorted(idx_test) if solo_prueba else range(len(df))
            st.session_state["idx_registro"] = int(np.random.choice(list(universo)))
    with col_a:
        idx = st.slider("Registro", 0, len(df) - 1, key="idx_registro")

    fila = df.iloc[[idx]]
    real = float(fila[nk.OBJETIVO].iloc[0])
    es_prueba = idx in idx_test

    st.info(
        f"Registro **{idx}** · {fila['dia_nombre'].iloc[0]} a las "
        f"{int(fila['hora'].iloc[0]):02d}:00 · CO(GT) medido: **{real:.2f} mg/m³** · "
        + ("🧪 conjunto de **prueba**" if es_prueba else "📚 conjunto de **entrenamiento**")
    )
    if not es_prueba:
        st.caption(
            "⚠️ Este registro se usó para entrenar: los modelos ya vieron su respuesta, "
            "así que el acierto aquí es optimista y no refleja lo que harían con datos "
            "nuevos. Marca la casilla de arriba y pulsa el dado para ver solo registros "
            "de prueba."
        )

    preds_reg = nk.predecir(modelos, fila, seleccionados)
    mostrar_predicciones(preds_reg, real=real)
    mostrar_veredicto([float(np.ravel(v)[0]) for v in preds_reg.values()])

    n_real = nk.interpretar(real)
    st.caption(
        f"Según el valor **realmente medido** ({real:.2f} mg/m³), el veredicto sería "
        f"«{n_real['icono']} {n_real['veredicto']}»."
    )

    st.divider()
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("**Lecturas de este registro**")
        st.dataframe(
            fila[nk.SENSORES + nk.AMBIENTALES].T.rename(columns={idx: "Valor"}),
            use_container_width=True,
        )
    with col2:
        st.markdown("**Real vs predicho**")
        comparacion = pd.DataFrame(
            {
                "Serie": ["Valor real"] + list(preds_reg),
                "CO(GT)": [real] + [float(np.ravel(v)[0]) for v in preds_reg.values()],
            }
        )
        fig = px.bar(
            comparacion,
            x="Serie",
            y="CO(GT)",
            color="Serie",
            text_auto=".2f",
            color_discrete_map={**nk.COLORES, "Valor real": "#424242"},
            labels={"CO(GT)": "CO(GT) (mg/m³)"},
        )
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ==========================================================================
# PESTAÑA 3 — Predicción por lotes
# ==========================================================================
with tab_lote:
    st.subheader("Sube un CSV y predice muchos registros de una vez")
    st.markdown(
        "El archivo puede venir en dos formatos:\n"
        "1. Con las **18 variables derivadas** ya calculadas (como `AirQualityUCI_limpio.csv`).\n"
        "2. Con las columnas **en crudo**: los 5 sensores `PT08.S*`, `T`, `RH`, `AH`, "
        "más `hora` (0-23) y `day_of_week` (0 = lunes)."
    )

    plantilla = df.head(20)[nk.SENSORES + nk.AMBIENTALES + ["hora", "day_of_week"]]
    st.download_button(
        "⬇️ Descargar plantilla de ejemplo (20 filas)",
        plantilla.to_csv(index=False).encode("utf-8"),
        file_name="plantilla_lote.csv",
        mime="text/csv",
    )

    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        try:
            crudo = pd.read_csv(archivo)
            X_lote = nk.preparar_lote(crudo)
        except Exception as e:
            st.error(f"No se pudo procesar el archivo: {e}")
        else:
            st.success(f"{len(X_lote)} filas listas para predecir.")
            resultado = crudo.copy()
            for nombre, pred in nk.predecir(modelos, X_lote, seleccionados).items():
                resultado[f"Predicción — {nombre}"] = pred

            columnas_pred = [c for c in resultado.columns if c.startswith("Predicción")]
            st.dataframe(resultado, use_container_width=True, height=360)

            st.markdown("**Resumen de las predicciones**")
            st.dataframe(resultado[columnas_pred].describe().T, use_container_width=True)

            if nk.OBJETIVO in crudo.columns:
                st.markdown("**Error frente al CO(GT) real incluido en el archivo**")
                errores = pd.DataFrame(
                    {
                        "Modelo": [c.replace("Predicción — ", "") for c in columnas_pred],
                        "MAE": [
                            float(np.mean(np.abs(crudo[nk.OBJETIVO] - resultado[c])))
                            for c in columnas_pred
                        ],
                        "RMSE": [
                            float(np.sqrt(np.mean((crudo[nk.OBJETIVO] - resultado[c]) ** 2)))
                            for c in columnas_pred
                        ],
                    }
                )
                st.dataframe(errores, use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Descargar resultados",
                resultado.to_csv(index=False).encode("utf-8"),
                file_name="predicciones_CO.csv",
                mime="text/csv",
                type="primary",
            )

# ==========================================================================
# PESTAÑA 4 — Comparación de modelos
# ==========================================================================
with tab_comparar:
    st.subheader("Métricas sobre el conjunto de prueba (20 % = 1.535 registros)")
    st.caption(
        "Recalculadas en vivo reproduciendo el mismo split de los cuadernos "
        "(`train_test_split(test_size=0.2, random_state=42)`)."
    )

    vista = tabla_metricas.copy().sort_values("R² prueba", ascending=False)
    st.dataframe(
        vista,
        use_container_width=True,
        hide_index=True,
        column_config={
            c: st.column_config.NumberColumn(c, format="%.4f")
            for c in ["MAE", "MSE", "RMSE", "R² entrenamiento", "R² prueba", "Brecha ΔR²"]
        }
        | {"MAPE (%)": st.column_config.NumberColumn("MAPE (%)", format="%.2f")},
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            vista,
            x="Modelo",
            y="R² prueba",
            color="Modelo",
            text_auto=".4f",
            color_discrete_map=nk.COLORES,
            title="R² en prueba (mayor es mejor)",
        )
        fig.update_layout(height=360, showlegend=False, yaxis_range=[0.8, 0.92])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(
            vista.melt(id_vars="Modelo", value_vars=["MAE", "RMSE"]),
            x="Modelo",
            y="value",
            color="variable",
            barmode="group",
            text_auto=".3f",
            labels={"value": "Error (mg/m³)", "variable": "Métrica"},
            title="Error de predicción (menor es mejor)",
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "La **brecha ΔR²** (R² entrenamiento − R² prueba) mide el sobreajuste: "
        "cuanto más pequeña, mejor generaliza el modelo."
    )

    st.divider()
    st.subheader("Diagnóstico sobre el conjunto de prueba")
    diag = nk.predicciones_prueba()
    diag = diag[diag["Modelo"].isin(seleccionados)]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Real vs predicho**")
        fig = px.scatter(
            diag,
            x="Real",
            y="Predicho",
            color="Modelo",
            opacity=0.35,
            color_discrete_map=nk.COLORES,
            labels={"Real": "CO(GT) real (mg/m³)", "Predicho": "CO(GT) predicho (mg/m³)"},
        )
        lim = [float(diag["Real"].min()), float(diag["Real"].max())]
        fig.add_trace(
            go.Scatter(
                x=lim, y=lim, mode="lines", name="Predicción perfecta",
                line=dict(color="red", dash="dash"),
            )
        )
        fig.update_layout(height=420, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**Distribución de los residuos (real − predicho)**")
        fig = px.histogram(
            diag,
            x="Residuo",
            color="Modelo",
            nbins=60,
            opacity=0.6,
            barmode="overlay",
            color_discrete_map=nk.COLORES,
            labels={"Residuo": "Residuo (mg/m³)"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(height=420, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Importancia de las variables")
    st.caption(
        "Solo para los modelos basados en árboles. El SVR no expone "
        "importancias directas: habría que calcular permutation_importance, demasiado "
        "costoso para una app interactiva."
    )
    imp = nk.importancias()
    arboles = [m for m in seleccionados if m in imp["Modelo"].unique()]
    if arboles:
        imp = imp[imp["Modelo"].isin(arboles)]
        orden = (
            imp.groupby("Variable")["Importancia"].mean().sort_values(ascending=True).index.tolist()
        )
        fig = px.bar(
            imp,
            x="Importancia",
            y="Variable",
            color="Modelo",
            barmode="group",
            orientation="h",
            color_discrete_map=nk.COLORES,
            category_orders={"Variable": orden},
        )
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Selecciona Bosques Aleatorios o Árbol de Decisión para ver esta sección.")

    with st.expander("Hiperparámetros de cada modelo"):
        st.dataframe(
            pd.DataFrame(
                {"Modelo": list(nk.CONFIGURACION), "Configuración": list(nk.CONFIGURACION.values())}
            ),
            use_container_width=True,
            hide_index=True,
        )

# ==========================================================================
# PESTAÑA 5 — Exploración de datos
# ==========================================================================
with tab_datos:
    st.subheader("El dataset que alimentó a los modelos")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{len(df):,}".replace(",", "."))
    c2.metric("Variables predictoras", len(nk.FEATURES))
    c3.metric("CO(GT) medio", f"{df[nk.OBJETIVO].mean():.2f} mg/m³")
    c4.metric("Rango de CO(GT)", f"{df[nk.OBJETIVO].min():.1f} – {df[nk.OBJETIVO].max():.1f}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Patrón horario del CO medido**")
        por_hora = df.groupby("hora")[nk.OBJETIVO].agg(["mean", "std"]).reset_index()
        fig = px.line(
            por_hora, x="hora", y="mean", markers=True,
            labels={"hora": "Hora del día", "mean": "CO(GT) promedio (mg/m³)"},
        )
        fig.update_traces(line_color="#2E7D32")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Los dos picos (mañana y tarde) justifican el uso de dos armónicos de hora.")

    with col2:
        st.markdown("**Patrón semanal del CO medido**")
        por_dia = (
            df.groupby("day_of_week")[nk.OBJETIVO].mean().reset_index().assign(
                Día=lambda d: d["day_of_week"].map(dict(enumerate(nk.DIAS)))
            )
        )
        fig = px.bar(
            por_dia, x="Día", y=nk.OBJETIVO, text_auto=".2f",
            labels={nk.OBJETIVO: "CO(GT) promedio (mg/m³)"},
        )
        fig.update_traces(marker_color="#1565C0")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Menor concentración el fin de semana: la fuente principal es el tráfico.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Distribución del CO(GT) frente a los umbrales**")
        fig = px.histogram(df, x=nk.OBJETIVO, nbins=60, labels={nk.OBJETIVO: "CO(GT) (mg/m³)"})
        fig.update_traces(marker_color="#2E7D32")
        for corte, texto in [(2, "leve"), (4, "guía OMS 24 h"), (10, "límite 8 h")]:
            fig.add_vline(
                x=corte, line_dash="dash", line_color="#C62828",
                annotation_text=texto, annotation_position="top",
            )
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
        reparto = pd.cut(
            df[nk.OBJETIVO], [0, 2, 4, 10, np.inf], right=False,
            labels=[n["nivel"] for n in nk.NIVELES],
        ).value_counts(normalize=True).reindex([n["nivel"] for n in nk.NIVELES]) * 100
        st.caption(
            "Reparto del dataset: "
            + " · ".join(f"**{k}** {v:.1f} %" for k, v in reparto.items())
        )
    with col2:
        st.markdown("**Relación sensor-objetivo**")
        sensor = st.selectbox("Variable", nk.SENSORES + nk.AMBIENTALES, index=0)
        muestra = df.sample(min(2500, len(df)), random_state=42)
        fig = px.scatter(
            muestra, x=sensor, y=nk.OBJETIVO, opacity=0.35,
            labels={nk.OBJETIVO: "CO(GT) (mg/m³)"},
        )
        # Recta de ajuste calculada con numpy, para no depender de statsmodels.
        pendiente, intercepto = np.polyfit(df[sensor], df[nk.OBJETIVO], 1)
        xs = np.array([df[sensor].min(), df[sensor].max()])
        fig.add_trace(
            go.Scatter(
                x=xs, y=pendiente * xs + intercepto, mode="lines",
                name="Tendencia lineal", line=dict(color="red"),
            )
        )
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Correlación de {sensor} con CO(GT): **{df[sensor].corr(df[nk.OBJETIVO]):.3f}**")

    st.markdown("**Matriz de correlación**")
    corr = df[nk.SENSORES + nk.AMBIENTALES + [nk.OBJETIVO]].corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver estadísticos descriptivos"):
        st.dataframe(stats, use_container_width=True)
    with st.expander("Ver los datos en crudo"):
        st.dataframe(df.head(500), use_container_width=True, height=400)
