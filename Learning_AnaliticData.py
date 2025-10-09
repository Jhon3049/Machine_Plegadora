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
    "4.50mm | CAL 3/16in": 4.50, "4.75mm | 3/16in": 4.75,
    "6.00mm | CAL 1/4in": 6.00, "6.35mm | 1/4in": 6.35,
    "7.94mm | CAL 5/16in": 7.94, "8.00mm | 5/16in": 8.00
}

nombre_acero = {
    "Acero Inoxidable 430 (magnético)": 430,
    "Acero 1020 COLD ROLLED": 10201,
    "Acero 1020 HOT ROLLED": 10200,
    "Acero inoxidable 304": 304
}

# === Título principal ===
st.title("🧠 Sistema Predictivo de Ángulos para Plegadora CNC")
st.write("Predice el valor de **Y (mm)** para el doblado de láminas según los parámetros seleccionados. Modelo basado en *Machine Learning (Random Forest)*.")

# === Entradas de usuario ===
st.header("📥 Parámetros de Entrada")

angulo = st.number_input("Ángulo deseado (°)", min_value=0, max_value=110, step=1)
longitud = st.number_input("Longitud de plegado (mm)", min_value=0.0, max_value=2500.0, step=1.0)
radio = st.selectbox("Radio o Apertura de matriz (V)", list(radio_v_map.keys()))
v = radio_v_map[radio]
espesor_t = st.selectbox("Espesor de la lámina", list(calibre_milimetros.keys()))
espesor = calibre_milimetros[espesor_t]
tipo_acero = st.selectbox("Tipo de acero", list(nombre_acero.keys()))
cdg_data = nombre_acero[tipo_acero]

# === Botón de ejecución ===
if st.button("🔮 Predecir Valor Y (mm)"):
    ruta_excel = Path(r"data_base.xlsx")

    if not ruta_excel.exists():
        st.error("❌ No se encontró el archivo de datos. Verifica la ruta del Excel.")
    else:
        # === Cargar datos ===
        data = pd.read_excel(ruta_excel)

        # Verificación de columnas esperadas
        columnas_necesarias = {"angulo", "v", "s", "l", "acero", "y"}
        if not columnas_necesarias.issubset(data.columns):
            st.error("⚠️ El archivo de Excel no contiene todas las columnas necesarias.")
        else:
            # === Preparación de datos ===
            X = data[["angulo", "v", "s", "l", "acero"]]
            y = data["y"]

            columnas_num = ["angulo", "v", "s", "l"]
            columnas_cat = ["acero"]

            # === Preprocesamiento ===
            preprocesador = ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_cat)
            ], remainder="passthrough")

            # === Modelo ===
            modelo = RandomForestRegressor(
                n_estimators=1000,  
                random_state=42,
                n_jobs=-1  # Aprovecha todos los núcleos del CPU
            )

            # === Pipeline completo ===
            pipeline = Pipeline([
                ("preprocesamiento", preprocesador),
                ("modelo", modelo)
            ])

            # === División de datos ===
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # === Entrenamiento ===
            with st.spinner("Entrenando modelo... ⏳"):
                pipeline.fit(X_train, y_train)

            # === Evaluación ===
            pred_test = pipeline.predict(X_test)
            mae = mean_absolute_error(y_test, pred_test)
            r2 = r2_score(y_test, pred_test)

            # === Nueva predicción ===
            nuevo = pd.DataFrame({
                "angulo": [angulo],
                "v": [v],
                "s": [espesor],
                "l": [longitud],
                "acero": [cdg_data]
            })

            pred_y = pipeline.predict(nuevo)[0]

            # === Resultados ===
            st.success(f"✅ Predicción del valor **Y**: {pred_y:.2f} mm")
            st.info(f"📊 Error medio absoluto (MAE): {mae:.3f}")

            st.info(f"📈 Coeficiente de determinación (R²): {r2:.3f}")






