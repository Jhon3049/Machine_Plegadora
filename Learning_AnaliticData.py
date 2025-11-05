# === Interfaz y Machine Learning Predictivo ===
import os
import base64
import requests
from io import BytesIO
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = "Jhon3049/Machine_Plegadora"
RUTA_EXCEL_REPO = "data_base.xlsx"  # ruta dentro del repo
BRANCH = "main"

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

def obtener_excel_desde_github():
    """Descarga el Excel actual desde el repositorio de GitHub."""
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{BRANCH}/{RUTA_EXCEL_REPO}"
    try:
        data = pd.read_excel(url)
        return data
    except Exception as e:
        st.error(f"⚠️ Error al leer el archivo desde GitHub: {e}")
        return pd.DataFrame()  # DataFrame vacío si falla


def subir_excel_a_github(df):
    """Sube el archivo Excel actualizado directamente al repositorio GitHub."""
    try:
        # Convertir el DataFrame a bytes Excel
        buffer = BytesIO()
        df.to_excel(buffer, index=False)
        content = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Obtener SHA del archivo actual
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{RUTA_EXCEL_REPO}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            sha = response.json()["sha"]
        else:
            sha = None

        # Crear/actualizar archivo
        data = {
            "message": "Actualización automática desde Streamlit",
            "content": content,
            "branch": BRANCH
        }
        if sha:
            data["sha"] = sha

        put_response = requests.put(url, headers=headers, json=data)

        if put_response.status_code in [200, 201]:
            st.success("☁️ Archivo actualizado correctamente en GitHub.")
        else:
            st.error(f"❌ Error al subir archivo a GitHub: {put_response.json()}")

    except Exception as e:
        st.error(f"⚠️ Error durante la subida a GitHub: {e}")

# === Título ===
st.title("🧠 Sistema Predictivo de Ángulos para Plegadora CNC")
st.write("Predice el valor de **Y (mm)** para el doblado de láminas según los parámetros seleccionados. Modelo basado en *Machine Learning (Random Forest)*.")

tabs = st.tabs(["📐 Cálculo de Ángulo y Datos Técnicos", "⚙️ Ajuste de la Máquina"])

# ============================================================
# === TAB 2: AJUSTE ===
# ============================================================
with tabs[1]:
    st.header("⚙️ Ajuste de Punzones")

    st.info("""
    **Condiciones estándar de calibración:**
    - Probeta: 50 mm × 100 mm  
    - Material: **COLD ROLLED CAL 14**  
    - Dado: **V = 26**  
    - Punzón: **200 mm**  
    - Ángulo objetivo: **90°**  
    - Valor de referencia: **Y = 92.70 mm**
    """)

    st.caption("La calibración se realiza con las probetas de 50mm*100mm COLD ROLLED de CAL14, Con el dado V=26, Con el Punzon de 200 mm. SE DEBE LLEVAR A 90° EN Y= 92.70mm")

    st.markdown("---")

    st.subheader("📏 Valores de Alineación de la Matriz")

    st.markdown("""
    <style>
    .table-container {
    display: flex;
    justify-content: center;
    margin-top: 10px;
    }

    .pretty-table {
    border-collapse: collapse;
    width: 65%;
    background-color: #ffffff;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    font-family: "Segoe UI", sans-serif;
    animation: fadeIn 0.6s ease-in-out;
    }

    .pretty-table th {
    background: linear-gradient(90deg, #1E88E5, #42A5F5);
    color: white;
    text-align: center;
    padding: 12px;
    font-size: 16px;
    font-weight: 600;
    }

    .pretty-table td {
    text-align: center;
    padding: 10px;
    font-size: 15px;
    color: #333;
    border-bottom: 1px solid #f0f0f0;
    }

    .pretty-table tr:nth-child(even) {
    background-color: #f9f9f9;
    }

    .pretty-table tr:hover {
    background-color: #e3f2fd;
    transform: scale(1.01);
    transition: all 0.2s ease-in-out;
    }

    @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
    }
    </style>

    <div class="table-container">
    <table class="pretty-table">
        <tr>
        <th>V (mm)</th>
        <th>Valores en Y (mm)</th>
        </tr>
        <tr><td>6</td><td>85.5</td></tr>
        <tr><td>8</td><td>88.0</td></tr>
        <tr><td>12</td><td>90.8</td></tr>
        <tr><td>15</td><td>93.0</td></tr>
        <tr><td>20</td><td>95.5</td></tr>
        <tr><td>26</td><td>98.5</td></tr>
        <tr><td>37</td><td>105.5</td></tr>
        <tr><td>50</td><td>113.0</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### ⚙️ Valores Avanzados de la Máquina")
st.caption("Parámetros de configuración y calibración del controlador **E21 ESTUN**.")

st.divider()

# ======================================================
# 🧭 SECCIÓN CONST
# ======================================================
st.subheader("🔩 CONST — Configuración Base")
st.info("""
**Parámetros actuales:**
- Unidad de medida: **mm**  (`0 = mm`, `1 = inch`)
- Idioma: **English**  (`1 = English`, `0 = 中文`)
- Tiempo de liberación (*Release Time*): **0.30 s**
- Tiempo de pulso (*Pulse Time*): **0.200 s**
- Versión de firmware: **V1.17**
""")
st.caption("Estos valores corresponden a la pantalla de configuración *CONST* del controlador **E21 ESTUN**.")

st.divider()

# ======================================================
# ⚙️ SYSTEM PARA
# ======================================================
st.subheader("🧠 SYSTEM PARA — Parámetros del Sistema")
st.info("""
**Configuración actual:**
- X-Digits: **1**
- Y-Digits: **2**
- X-Safe: **10.0**
- Y-Safe: **5.0**
- Step Delay: **0.50 s**
- Count Select: **0**
- LDP Enable: **0**
- Trans. Select: **0**
""")
st.caption("Estos valores pertenecen al menú *SYSTEM PARA* del controlador **E21 ESTUN**.")
st.divider()

# ======================================================
# 🔧 X AXIS PARA
# ======================================================
st.subheader("🛠️ X AXIS PARA — Parámetros del Eje X")
st.info("""
**Configuración actual:**
- X_Enable: **1**
- Encoder Dir: **0**
- Teach. En: **1**
- Ref. Pos: **500.0**
- X-Min: **10.0**
- X-Max: **500.0**
- MF: **1500**
- DF: **10**
- Stop Dis.: **1.0**
- Tolerance: **0.200**
- Overrun En.: **1**
- Over. Dis.: **1.0**
- Repeat Enable: **1**
- Repeat Time: **0.50**
- Mute Dis.: **10.0**
- Stop Time: **0.50**
- OT Time: **0.50**
- Drive Mode: **1**
- High Freq.: **100%**
- Low Freq.: **10%**
""")
st.caption("Estos valores pertenecen al menú *X AXIS PARA* del controlador **E21 ESTUN**.")
st.divider()

st.subheader("🛠️ Y AXIS PARA — Parámetros del Eje Y")
st.info("""
**Configuración actual:**
- X_Enable: **1**
- Encoder Dir: **1**
- Teach. En: **1**
- Ref. Pos: **150.0**
- X-Min: **60.0**
- X-Max: **150.0**
- MF: **4105**
- DF: **10**
- Stop Dis.: **1.0**
- Tolerance: **0.020**
- Overrun En.: **1**
- Over. Dis.: **0.10**
- Repeat Enable: **1**
- Repeat Time: **0.50**
- Mute Dis.: **2.0**
- Stop Time: **0.50**
- OT Time: **0.50**
- Drive Mode: **1**
- High Freq.: **100%**
- Low Freq.: **10%**
""")
st.caption("Estos valores pertenecen al menú *Y AXIS PARA* del controlador **E21 ESTUN**.")
st.divider()

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

                pred_y = round(pipeline.predict(pd.DataFrame({
                    "angulo": [angulo],
                    "v": [v],
                    "s": [espesor],
                    "l": [longitud],
                    "acero": [cdg_data]
                }))[0], 2)

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
        if colB.button("🔧 Corregir y registrar nuevos valores"):
            st.session_state["accion"] = "corregir"

    # === ACCIÓN DE CORREGIR ===
    if st.session_state.get("accion") == "corregir":
        st.warning("✏️ Ingrese los valores reales medidos del ángulo (°) y de Y (mm):")

        nuevo_angulo = st.number_input(
            "Valor real del ángulo (°)",
            min_value=1, max_value=110, step=1,
            key="angulo_real_input"
        )

        nuevo_y = st.number_input(
            "Valor real de Y (mm)",
            min_value=0.00, max_value=200.00, step=0.01,
            key="y_real_input"
        )

        guardar_correccion = st.button("💾 Guardar corrección", key="guardar_corr")

        if guardar_correccion:
            try:
                # Recuperar parámetros previos
                if "parametros" in st.session_state:
                    angulo, v, espesor, longitud, cdg_data = st.session_state["parametros"]
                    pred_y = st.session_state["pred_y"]
                else:
                    st.error("❌ No se encontraron los parámetros previos. Calcula de nuevo el valor Y.")
                    st.stop()

                # Crear nuevo registro
                nuevo_dato_corr = pd.DataFrame({
                    "angulo": [nuevo_angulo],
                    "v": [v],
                    "s": [espesor],
                    "l": [longitud],
                    "acero": [cdg_data],
                    "y": [nuevo_y]
                })

                # Leer, concatenar y subir
                data = obtener_excel_desde_github()
                data = pd.concat([data, nuevo_dato_corr], ignore_index=True)
                subir_excel_a_github(data)

                # Confirmación visual
                st.success("✅ Corrección registrada y actualizada en GitHub.")
                st.dataframe(nuevo_dato_corr)

                st.session_state["mostrar_botones"] = False
                st.session_state["accion"] = None

            except Exception as e:
                st.error(f"⚠️ Error al guardar la corrección: {e}")


