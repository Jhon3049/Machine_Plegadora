# === Interfaz y Machine Learning Predictivo ===
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path

# === Ruta del archivo Excel ===
ruta_excel = Path(r"data_base.xlsx")
# === Diccionarios Base ===
radio_v_map = {
    "1.0mm | V6mm": 6, "1.3mm | V8mm": 8, "2.0mm | V12mm": 12, "2.7mm | V15mm": 15,
    "3.3mm | V20mm": 20, "4.2mm | V26mm": 26, "5.8mm | V37mm": 37, "8.3mm | V50mm": 50
}

calibre_milimetros = {
    "0.80mm | CAL 20": 0.80, "0.85mm | CAL 20": 0.85, "0.90mm | CAL 20": 0.90,
    "1.00mm | CAL 19": 1.00, "1.10mm | CAL 18": 1.10, "1.15mm | CAL 18": 1.15,
    "1.20mm | CAL 18": 1.20, "1.40mm | CAL 16": 1.40, "1.45mm | CAL 16": 1.45,
    "1.50mm | CAL 16": 1.50, "1.80mm | CAL 14": 1.80, "1.85mm | CAL 14": 1.85,
    "1.90mm | CAL 14": 1.90, "2.00mm | CAL 13": 2.00, "2.30mm | CAL -": 2.30,
    "2.50mm | CAL 12": 2.50, "3.00mm | 1/8in": 3.00, "4.00mm | CAL 8": 4.00,
    "4.50mm | 3/16in": 4.50, "4.75mm | 3/16in": 4.75,
    "6.00mm | 1/4in": 6.00, "6.35mm | 1/4in": 6.35,
    "7.94mm | 5/16in": 7.94, "8.00mm | 5/16in": 8.00
}

nombre_acero = {
    "Acero Inoxidable 430 (magnético)": 430,
    "Acero 1020 COLD ROLLED": 10201,
    "Acero 1020 HOT ROLLED": 10200,
    "Acero inoxidable 304": 304
}

relacion_H_V = {6: "5", 8: "6", 12: "9", 15: "12", 20: "15", 26: "18", 37: "25", 50: "36"}
alineacion_m = {6: "85.5", 8: "88", 12: "90.8", 15: "93", 20: "95.5", 26: "98.5", 37: "105.5", 50: "113"}

# === Título ===
st.title("🧠 Sistema Predictivo de Ángulos para Plegadora CNC")
st.write("Predice el valor de **Y (mm)** para el doblado de láminas según los parámetros seleccionados. Modelo basado en *Machine Learning (Random Forest)*.")

tabs = st.tabs(["📐 Cálculo de Ángulo y Datos Técnicos", "⚙️ Ajuste de la Máquina"])

# ============================================================
# === TAB 1: CÁLCULO PRINCIPAL ===
# ============================================================
with tabs[0]:
    st.header("📊 Cálculo Predictivo y Datos Técnicos")
    st.header("📥 Parámetros de Entrada")
    # === Entradas de usuario ===

    angulo = st.number_input("Ángulo deseado (°)", min_value=0, max_value=110, step=1)
    longitud = st.number_input("Longitud de plegado (mm)", min_value=0.0, max_value=2500.0, step=1.0)
    radio = st.selectbox("Radio o Apertura de la matriz (V)", list(radio_v_map.keys()))
    v = radio_v_map[radio]
    h = relacion_H_V[v]
    ali = alineacion_m[v]
    espesor_t = st.selectbox("Espesor de la lámina", list(calibre_milimetros.keys()))
    espesor = calibre_milimetros[espesor_t]
    tipo_acero = st.selectbox("Tipo de acero", list(nombre_acero.keys()))
    cdg_data = nombre_acero[tipo_acero]

    if angulo == 0 or longitud == 0:
        st.warning("⚠️ Los valores no pueden ser cero. Ingresa valores válidos.")
        st.stop()  # Detiene la ejecución del resto del código

    # === Cálculo del modelo ===
    if st.button("🔮 Predecir Valor Y (mm)"):
        st.session_state["mostrar_botones"] = True
        st.session_state["accion"] = None  # reset acción previa

        if not ruta_excel.exists():
            st.error("❌ No se encontró el archivo de datos.")
        else:
            data = pd.read_excel(ruta_excel)

            columnas_necesarias = {"angulo", "v", "s", "l", "acero", "y"}
            if not columnas_necesarias.issubset(data.columns):
                st.error("⚠️ Faltan columnas en el Excel.")
            else:
                # === Entrenamiento y predicción ===
                X = data[["angulo", "v", "s", "l", "acero"]]
                y = data["y"]

                preprocesador = ColumnTransformer([
                    ("cat", OneHotEncoder(handle_unknown="ignore"), ["acero"])
                ], remainder="passthrough")

                modelo = RandomForestRegressor(
                    n_estimators=1000, random_state=42, n_jobs=-1
                )

                pipeline = Pipeline([
                    ("preprocesamiento", preprocesador),
                    ("modelo", modelo)
                ])

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                with st.spinner("Entrenando modelo... ⏳"):
                    pipeline.fit(X_train, y_train)

                # === Evaluación ===
                pred_test = pipeline.predict(X_test)
                mae = mean_absolute_error(y_test, pred_test)
                r2 = r2_score(y_test, pred_test)

                pred_y = pipeline.predict(pd.DataFrame({
                    "angulo": [angulo],
                    "v": [v],
                    "s": [espesor],
                    "l": [longitud],
                    "acero": [cdg_data]
                }))[0]

                st.session_state["pred_y"] = pred_y
                st.session_state["parametros"] = (angulo, v, espesor, longitud, cdg_data)

                st.success(f"✅ Valor Y predicho: {pred_y:.2f} mm")
                st.info(f"📊 Error medio absoluto (MAE): {mae:.3f}")
                st.info(f"📈 Coeficiente de determinación (R²): {r2:.3f}")

                # === Mostrar resultados técnicos ===
                st.markdown("---")

                st.subheader("📊 Resultados Técnicos del Plegado")

                p=round((1.42*42*(espesor**2)*longitud)/(1000*v),2)

                st.markdown(
                    """
                    <style>
                    .metric-card {
                        background-color: #f7f9fb;
                        padding: 20px;
                        border-radius: 15px;
                        box-shadow: 0px 3px 8px rgba(0,0,0,0.1);
                        text-align: center;
                        font-family: 'Segoe UI';
                    }
                    .metric-value {
                        font-size: 28px;
                        font-weight: bold;
                        color: #1E88E5;
                    }
                    .metric-label {
                        font-size: 16px;
                        color: #555;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # Colocar en tres columnas
                col1, col2, col3 = st.columns(3)

                with col3:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{h} mm</div>
                        <div class='metric-label'>Longitud minima de pliegue (H)</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{p} Ton</div>
                        <div class='metric-label'>Presión estimada (P)</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{ali} mm</div>
                        <div class='metric-label'>Alineación de matriz</div>
                    </div>
                    """, unsafe_allow_html=True)


    # === Botones Confirmar / Corregir ===
    if st.session_state.get("mostrar_botones"):
        st.markdown("---")
        st.subheader("⚙️ ¿Deseas registrar este resultado?")

        colA, colB = st.columns(2)
        if colA.button("✅ Confirmar resultado correcto"):
            st.session_state["accion"] = "confirmar"
        if colB.button("🔧 Corregir y registrar nuevo valor"):
            st.session_state["accion"] = "corregir"

    # === Acción de Confirmar ===
    if st.session_state.get("accion") == "confirmar":
        angulo, v, espesor, longitud, cdg_data = st.session_state["parametros"]
        pred_y = st.session_state["pred_y"]

        nuevo_dato = pd.DataFrame({
            "angulo": [angulo],
            "v": [v],
            "s": [espesor],
            "l": [longitud],
            "acero": [cdg_data],
            "y": [pred_y]
        })

        data = pd.read_excel(ruta_excel)
        data = pd.concat([data, nuevo_dato], ignore_index=True)
        data.to_excel(ruta_excel, index=False)

        st.success("✅ Resultado confirmado y guardado en la base de datos.")
        st.session_state["mostrar_botones"] = False
        st.session_state["accion"] = None

       # === Acción de Corregir ===
    if st.session_state.get("accion") == "corregir":
        st.warning("✏️ Ingrese el valor real medido del angulo(°):")
        nuevo_angulo = st.number_input("Valor real del angulo(°)", min_value=0, max_value=110, step=1)

        if st.button("💾 Guardar corrección"):
            v, espesor, longitud, cdg_data, pred_y = st.session_state["parametros"]

            nuevo_dato_corr = pd.DataFrame({
                "angulo": [nuevo_angulo],
                "v": [v],
                "s": [espesor],
                "l": [longitud],
                "acero": [cdg_data],
                "y": [pred_y]
            })
            
            data = pd.read_excel(ruta_excel)
            data = pd.concat([data, nuevo_dato_corr], ignore_index=True)
            data.to_excel(ruta_excel, index=False)

            st.success("💾 Corrección registrada exitosamente.")
            st.session_state["mostrar_botones"] = False
            st.session_state["accion"] = None


