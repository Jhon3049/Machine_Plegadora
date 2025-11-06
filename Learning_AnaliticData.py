"""
Sistema Predictivo de Ángulos para Plegadora CNC
Versión Optimizada - Aplicación Industrial Profesional

Mejoras implementadas:
- Interfaz UX/UI profesional con diseño modular
- Validación robusta de datos de entrada
- Manejo de errores comprehensivo
- Código limpio y mantenible
- Arquitectura escalable
"""

import os
import base64
import requests
from io import BytesIO
from typing import Dict, Tuple, Optional
from pathlib import Path

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score


# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN Y CONSTANTES
# ═══════════════════════════════════════════════════════════════

class Config:
    """Configuración centralizada de la aplicación"""
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    GITHUB_REPO = "Jhon3049/Machine_Plegadora"
    RUTA_EXCEL_REPO = "data_base.xlsx"
    BRANCH = "main"
    RUTA_LOCAL = Path("data_base.xlsx")
    
    # Parámetros del modelo
    MODEL_ESTIMATORS = 1000
    MODEL_TEST_SIZE = 0.2
    MODEL_RANDOM_STATE = 42


class MaterialConstants:
    """Constantes de materiales y configuraciones técnicas"""
    
    RADIO_V_MAP = {
        "1.0mm | V6mm": 6, "1.3mm | V8mm": 8, "2.0mm | V12mm": 12, 
        "2.7mm | V15mm": 15, "3.3mm | V20mm": 20, "4.2mm | V26mm": 26, 
        "5.8mm | V37mm": 37, "8.3mm | V50mm": 50
    }
    
    CALIBRE_MILIMETROS = {
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
    
    NOMBRE_ACERO = {
        "Acero Inoxidable 430 (magnético)": 430,
        "Acero 1020 COLD ROLLED": 10201,
        "Acero 1020 HOT ROLLED": 10200,
        "Acero inoxidable 304": 304
    }
    
    RELACION_H_V = {6: 5, 8: 6, 12: 9, 15: 12, 20: 15, 26: 18, 37: 25, 50: 36}
    ALINEACION_M = {6: 85.5, 8: 88.0, 12: 90.8, 15: 93.0, 20: 95.5, 26: 98.5, 37: 105.5, 50: 113.0}


# ═══════════════════════════════════════════════════════════════
# UTILIDADES Y FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════

class DataManager:
    """Gestión de datos con GitHub"""
    
    @staticmethod
    def obtener_datos_github() -> pd.DataFrame:
        """Descarga el Excel desde GitHub con manejo robusto de errores"""
        url = f"https://raw.githubusercontent.com/{Config.GITHUB_REPO}/{Config.BRANCH}/{Config.RUTA_EXCEL_REPO}"
        
        try:
            data = pd.read_excel(url)
            if data.empty:
                st.warning("⚠️ El archivo descargado está vacío")
                return pd.DataFrame()
            return data
        except requests.RequestException as e:
            st.error(f"❌ Error de conexión con GitHub: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def subir_datos_github(df: pd.DataFrame) -> bool:
        """Sube el DataFrame actualizado a GitHub"""
        if df.empty:
            st.error("❌ No se puede subir un DataFrame vacío")
            return False
        
        if not Config.GITHUB_TOKEN:
            st.error("❌ Token de GitHub no configurado")
            return False
        
        try:
            # Convertir DataFrame a bytes
            buffer = BytesIO()
            df.to_excel(buffer, index=False)
            content = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            # Obtener SHA del archivo actual
            url = f"https://api.github.com/repos/{Config.GITHUB_REPO}/contents/{Config.RUTA_EXCEL_REPO}"
            headers = {"Authorization": f"token {Config.GITHUB_TOKEN}"}
            response = requests.get(url, headers=headers, timeout=10)
            
            sha = response.json().get("sha") if response.status_code == 200 else None
            
            # Actualizar archivo
            data = {
                "message": "Actualización automática desde Streamlit",
                "content": content,
                "branch": Config.BRANCH
            }
            if sha:
                data["sha"] = sha
            
            put_response = requests.put(url, headers=headers, json=data, timeout=10)
            
            if put_response.status_code in [200, 201]:
                st.success("✅ Datos actualizados correctamente en GitHub")
                return True
            else:
                st.error(f"❌ Error al actualizar: {put_response.json().get('message', 'Desconocido')}")
                return False
                
        except Exception as e:
            st.error(f"❌ Error durante la actualización: {str(e)}")
            return False


class Validator:
    """Validación de datos de entrada"""
    
    @staticmethod
    def validar_parametros(angulo: float, longitud: float, espesor: float, v: int) -> Tuple[bool, str]:
        """Valida los parámetros de entrada"""
        if angulo <= 0 or angulo > 180:
            return False, "⚠️ El ángulo debe estar entre 1° y 180°"
        
        if longitud <= 0:
            return False, "⚠️ La longitud debe ser mayor a 0 mm"
        
        if espesor <= 0:
            return False, "⚠️ El espesor debe ser mayor a 0 mm"
        
        if v not in MaterialConstants.RELACION_H_V:
            return False, "⚠️ Valor V no válido"
        
        # Validación técnica: H mínimo
        h_min = MaterialConstants.RELACION_H_V[v]
        if longitud < h_min:
            return False, f"⚠️ La longitud mínima para V={v}mm es {h_min}mm"
        
        return True, ""
    
    @staticmethod
    def validar_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
        """Valida la estructura del DataFrame"""
        columnas_necesarias = {"angulo", "v", "s", "l", "acero", "y"}
        
        if df.empty:
            return False, "El DataFrame está vacío"
        
        if not columnas_necesarias.issubset(df.columns):
            faltantes = columnas_necesarias - set(df.columns)
            return False, f"Faltan columnas: {', '.join(faltantes)}"
        
        return True, ""


class ModeloPredictor:
    """Modelo de Machine Learning para predicción"""
    
    def __init__(self):
        self.pipeline = None
        self.mae = None
        self.r2 = None
    
    def entrenar(self, data: pd.DataFrame) -> bool:
        """Entrena el modelo con los datos proporcionados"""
        try:
            X = data[["angulo", "v", "s", "l", "acero"]]
            y = data["y"]
            
            preprocesador = ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["acero"])
            ], remainder="passthrough")
            
            modelo = RandomForestRegressor(
                n_estimators=Config.MODEL_ESTIMATORS,
                random_state=Config.MODEL_RANDOM_STATE,
                n_jobs=-1
            )
            
            self.pipeline = Pipeline([
                ("preprocesamiento", preprocesador),
                ("modelo", modelo)
            ])
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=Config.MODEL_TEST_SIZE, random_state=Config.MODEL_RANDOM_STATE
            )
            
            self.pipeline.fit(X_train, y_train)
            
            # Evaluar modelo
            pred_test = self.pipeline.predict(X_test)
            self.mae = mean_absolute_error(y_test, pred_test)
            self.r2 = r2_score(y_test, pred_test)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error al entrenar el modelo: {str(e)}")
            return False
    
    def predecir(self, angulo: float, v: int, espesor: float, longitud: float, acero: int) -> Optional[float]:
        """Realiza una predicción"""
        if self.pipeline is None:
            st.error("❌ El modelo no ha sido entrenado")
            return None
        
        try:
            input_data = pd.DataFrame({
                "angulo": [angulo],
                "v": [v],
                "s": [espesor],
                "l": [longitud],
                "acero": [acero]
            })
            
            prediccion = self.pipeline.predict(input_data)[0]
            return round(prediccion, 2)
            
        except Exception as e:
            st.error(f"❌ Error en la predicción: {str(e)}")
            return None


# ═══════════════════════════════════════════════════════════════
# COMPONENTES DE INTERFAZ
# ═══════════════════════════════════════════════════════════════

def aplicar_estilos_personalizados():
    """Aplica estilos CSS personalizados para una interfaz profesional"""
    st.markdown("""
    <style>
    /* Esquema de colores industrial profesional */
    :root {
        --primary-color: #1E3A8A;
        --secondary-color: #3B82F6;
        --accent-color: #10B981;
        --warning-color: #F59E0B;
        --danger-color: #EF4444;
        --neutral-light: #F3F4F6;
        --neutral-dark: #1F2937;
    }
    
    /* Contenedor principal */
    .main {
        background-color: #F9FAFB;
    }
    
    /* Tarjetas de métricas mejoradas */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin: 10px 0;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.25);
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        margin-bottom: 8px;
    }
    
    .metric-label {
        font-size: 14px;
        color: rgba(255,255,255,0.9);
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    /* Tabla profesional */
    .professional-table {
        width: 100%;
        border-collapse: collapse;
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    
    .professional-table th {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 16px;
        font-weight: 600;
        text-align: left;
        font-size: 14px;
        letter-spacing: 0.5px;
    }
    
    .professional-table td {
        padding: 14px 16px;
        border-bottom: 1px solid #E5E7EB;
        font-size: 14px;
        color: #374151;
    }
    
    .professional-table tr:hover {
        background-color: #F3F4F6;
    }
    
    /* Encabezados de sección */
    .section-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        margin: 24px 0 16px 0;
        font-size: 20px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
    }
    
    /* Alertas personalizadas */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid var(--primary-color);
    }
    
    /* Botones mejorados */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        padding: 12px 24px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Inputs profesionales */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 2px solid #E5E7EB;
        transition: border-color 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)


def renderizar_metricas(h: float, presion: float, alineacion: float):
    """Renderiza las métricas técnicas con diseño profesional"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{alineacion} mm</div>
            <div class='metric-label'>Alineación de Matriz</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{presion} Ton</div>
            <div class='metric-label'>Presión Estimada</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{h} mm</div>
            <div class='metric-label'>Longitud Mínima (H)</div>
        </div>
        """, unsafe_allow_html=True)


def renderizar_tabla_alineacion():
    """Renderiza la tabla de valores de alineación"""
    tabla_html = """
    <table class='professional-table'>
        <thead>
            <tr>
                <th>V (mm)</th>
                <th>Valores en Y (mm)</th>
                <th>H Mínimo (mm)</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for v, ali in MaterialConstants.ALINEACION_M.items():
        h = MaterialConstants.RELACION_H_V[v]
        tabla_html += f"<tr><td>{v}</td><td>{ali}</td><td>{h}</td></tr>"
    
    tabla_html += "</tbody></table>"
    st.markdown(tabla_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# APLICACIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def inicializar_sesion():
    """Inicializa variables de sesión"""
    if "mostrar_botones" not in st.session_state:
        st.session_state.mostrar_botones = False
    if "accion" not in st.session_state:
        st.session_state.accion = None
    if "pred_y" not in st.session_state:
        st.session_state.pred_y = None
    if "parametros" not in st.session_state:
        st.session_state.parametros = None


def main():
    """Función principal de la aplicación"""
    
    # Configuración de página
    st.set_page_config(
        page_title="Sistema Predictivo Plegadora CNC",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    aplicar_estilos_personalizados()
    inicializar_sesion()
    
    # Encabezado principal
    st.markdown("<h1 style='text-align: center; color: #1E3A8A; margin-bottom: 10px;'>🏭 Sistema Predictivo de Plegadora CNC</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6B7280; font-size: 18px; margin-bottom: 30px;'>Predicción de valores Y mediante Machine Learning | Random Forest Algorithm</p>", unsafe_allow_html=True)
    
    # Pestañas principales
    tab1, tab2 = st.tabs(["📊 Cálculo Predictivo", "⚙️ Configuración de Máquina"])
    
    # ═══════════════════════════════════════════════════════════════
    # TAB 1: CÁLCULO PREDICTIVO
    # ═══════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("<div class='section-header'>📥 Parámetros de Entrada</div>", unsafe_allow_html=True)
        
        # Layout de entradas en dos columnas
        col1, col2 = st.columns(2)
        
        with col1:
            angulo = st.number_input(
                "🔺 Ángulo deseado (°)",
                min_value=1,
                max_value=180,
                value=90,
                step=1,
                help="Ángulo objetivo del doblado (1° - 180°)"
            )
            
            radio = st.selectbox(
                "📐 Radio/Apertura de matriz (V)",
                list(MaterialConstants.RADIO_V_MAP.keys()),
                help="Seleccione la apertura V de la matriz"
            )
            
            tipo_acero = st.selectbox(
                "🔩 Tipo de material",
                list(MaterialConstants.NOMBRE_ACERO.keys()),
                help="Material de la lámina a plegar"
            )
        
        with col2:
            longitud = st.number_input(
                "📏 Longitud de plegado (mm)",
                min_value=0.0,
                max_value=2500.0,
                value=100.0,
                step=1.0,
                help="Longitud total del doblez"
            )
            
            espesor_t = st.selectbox(
                "📊 Espesor de la lámina",
                list(MaterialConstants.CALIBRE_MILIMETROS.keys()),
                help="Calibre/espesor del material"
            )
        
        # Obtener valores numéricos
        v = MaterialConstants.RADIO_V_MAP[radio]
        espesor = MaterialConstants.CALIBRE_MILIMETROS[espesor_t]
        cdg_data = MaterialConstants.NOMBRE_ACERO[tipo_acero]
        h = MaterialConstants.RELACION_H_V[v]
        alineacion = MaterialConstants.ALINEACION_M[v]
        
        # Validación previa
        valido, mensaje = Validator.validar_parametros(angulo, longitud, espesor, v)
        
        if not valido:
            st.error(mensaje)
        
        st.markdown("---")
        
        # Botón de predicción
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predecir_btn = st.button("🔮 Calcular Valor Y", type="primary", use_container_width=True)
        
        if predecir_btn:
            if not valido:
                st.error("❌ Por favor corrija los errores antes de continuar")
                st.stop()
            
            with st.spinner("🔄 Entrenando modelo y realizando predicción..."):
                # Cargar datos
                data = DataManager.obtener_datos_github()
                
                if data.empty:
                    st.error("❌ No se pudieron cargar los datos de entrenamiento")
                    st.stop()
                
                # Validar estructura
                valido_df, mensaje_df = Validator.validar_dataframe(data)
                if not valido_df:
                    st.error(f"❌ Error en los datos: {mensaje_df}")
                    st.stop()
                
                # Entrenar y predecir
                modelo = ModeloPredictor()
                if not modelo.entrenar(data):
                    st.stop()
                
                pred_y = modelo.predecir(angulo, v, espesor, longitud, cdg_data)
                
                if pred_y is None:
                    st.stop()
                
                # Guardar en sesión
                st.session_state.pred_y = pred_y
                st.session_state.parametros = (angulo, v, espesor, longitud, cdg_data)
                st.session_state.mostrar_botones = True
                
                # Mostrar resultado principal
                st.markdown("<div class='section-header'>✅ Resultado de la Predicción</div>", unsafe_allow_html=True)
                
                col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
                with col_res2:
                    st.success(f"### 🎯 Valor Y predicho: **{pred_y:.2f} mm**")
                
                # Métricas del modelo
                col_met1, col_met2 = st.columns(2)
                with col_met1:
                    st.info(f"📊 **Error medio absoluto (MAE):** {modelo.mae:.3f} mm")
                with col_met2:
                    st.info(f"📈 **Coeficiente R²:** {modelo.r2:.3f}")
                
                st.markdown("---")
                
                # Cálculos técnicos
                presion = round((1.42 * 42 * (espesor ** 2) * longitud) / (1000 * v), 2)
                
                st.markdown("<div class='section-header'>🔧 Parámetros Técnicos del Doblado</div>", unsafe_allow_html=True)
                renderizar_metricas(h, presion, alineacion)
        
        # Botones de confirmación/corrección
        if st.session_state.get("mostrar_botones"):
            st.markdown("---")
            st.markdown("<div class='section-header'>💾 Registro de Resultado</div>", unsafe_allow_html=True)
            
            col_action1, col_action2 = st.columns(2)
            with col_action1:
                if st.button("✅ Confirmar resultado correcto", use_container_width=True):
                    st.session_state.accion = "confirmar"
                    st.success("✅ Resultado confirmado. No se realizaron cambios en la base de datos.")
                    st.session_state.mostrar_botones = False
            
            with col_action2:
                if st.button("🔧 Corregir y registrar valores reales", use_container_width=True):
                    st.session_state.accion = "corregir"
        
        # Proceso de corrección
        if st.session_state.get("accion") == "corregir":
            st.markdown("---")
            st.warning("✏️ **Ingrese los valores reales medidos:**")
            
            col_corr1, col_corr2 = st.columns(2)
            with col_corr1:
                nuevo_angulo = st.number_input(
                    "Ángulo real medido (°)",
                    min_value=1,
                    max_value=180,
                    value=angulo,
                    step=1,
                    key="angulo_real"
                )
            
            with col_corr2:
                nuevo_y = st.number_input(
                    "Valor Y real medido (mm)",
                    min_value=0.00,
                    max_value=200.00,
                    value=st.session_state.pred_y if st.session_state.pred_y else 0.0,
                    step=0.01,
                    key="y_real"
                )
            
            if st.button("💾 Guardar corrección", type="primary"):
                if "parametros" not in st.session_state or st.session_state.parametros is None:
                    st.error("❌ No se encontraron los parámetros. Por favor, realice una nueva predicción.")
                    st.stop()
                
                angulo_orig, v_orig, espesor_orig, longitud_orig, acero_orig = st.session_state.parametros
                
                nuevo_dato = pd.DataFrame({
                    "angulo": [nuevo_angulo],
                    "v": [v_orig],
                    "s": [espesor_orig],
                    "l": [longitud_orig],
                    "acero": [acero_orig],
                    "y": [nuevo_y]
                })
                
                data = DataManager.obtener_datos_github()
                if not data.empty:
                    data = pd.concat([data, nuevo_dato], ignore_index=True)
                    
                    if DataManager.subir_datos_github(data):
                        st.success("✅ Corrección registrada exitosamente")
                        st.dataframe(nuevo_dato, use_container_width=True)
                        st.session_state.mostrar_botones = False
                        st.session_state.accion = None
    
    # ═══════════════════════════════════════════════════════════════
    # TAB 2: CONFIGURACIÓN DE MÁQUINA
    # ═══════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("<div class='section-header'>⚙️ Configuración del Controlador E21 ESTUN</div>", unsafe_allow_html=True)
        
        # Condiciones de calibración
        st.info("""
        **🔧 Condiciones estándar de calibración:**
        - **Probeta:** 50 mm × 100 mm  
        - **Material:** COLD ROLLED CAL 14  
        - **Dado (V):** 26 mm  
        - **Punzón:** 200 mm  
        - **Ángulo objetivo:** 90°  
        - **Valor de referencia:** Y = 92.70 mm
        """)
        
        st.markdown("---")
        
        # Tabla de alineación
        st.markdown("<div class='section-header'>📐 Valores de Alineación de Matriz</div>", unsafe_allow_html=True)
        renderizar_tabla_alineacion()
        
        st.markdown("---")
        
        # Configuraciones avanzadas en acordeón
        with st.expander("🔧 CONST - Configuración Base", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                - **Unidad de medida:** mm (`0 = mm`, `1 = inch`)
                - **Idioma:** English (`1 = English`, `0 = 中文`)
                - **Tiempo de liberación:** 0.30 s
                """)
            with col2:
                st.markdown("""
                - **Tiempo de pulso:** 0.200 s
                - **Versión de firmware:** V1.17
                """)
        
        with st.expander("🧠 SYSTEM PARA - Parámetros del Sistema", expanded=False):
            df_system = pd.DataFrame({
                "Parámetro": ["X-Digits", "Y-Digits", "X-Safe", "Y-Safe", "Step Delay", "Count Select", "LDP Enable", "Trans. Select"],
                "Valor": ["1", "2", "10.0", "5.0", "0.50 s", "0", "0", "0"]
            })
            st.dataframe(df_system, use_container_width=True, hide_index=True)
        
        with st.expander("🛠️ X AXIS PARA - Parámetros del Eje X", expanded=False):
            df_x_axis = pd.DataFrame({
                "Parámetro": ["X_Enable", "Encoder Dir", "Teach. En", "Ref. Pos", "X-Min", "X-Max", 
                             "MF", "DF", "Stop Dis.", "Tolerance", "Overrun En.", "Over. Dis.",
                             "Repeat Enable", "Repeat Time", "Mute Dis.", "Stop Time", "OT Time",
                             "Drive Mode", "High Freq.", "Low Freq."],
                "Valor": ["1", "0", "1", "500.0", "10.0", "500.0",
                         "1500", "10", "1.0", "0.200", "1", "1.0",
                         "1", "0.50", "10.0", "0.50", "0.50",
                         "1", "100%", "10%"]
            })
            st.dataframe(df_x_axis, use_container_width=True, hide_index=True)
        
        with st.expander("🛠️ Y AXIS PARA - Parámetros del Eje Y", expanded=False):
            df_y_axis = pd.DataFrame({
                "Parámetro": ["Y_Enable", "Encoder Dir", "Teach. En", "Ref. Pos", "Y-Min", "Y-Max", 
                             "MF", "DF", "Stop Dis.", "Tolerance", "Overrun En.", "Over. Dis.",
                             "Repeat Enable", "Repeat Time", "Mute Dis.", "Stop Time", "OT Time",
                             "Drive Mode", "High Freq.", "Low Freq."],
                "Valor": ["1", "1", "1", "150.0", "60.0", "150.0",
                         "4105", "10", "1.0", "0.020", "1", "0.10",
                         "1", "0.50", "2.0", "0.50", "0.50",
                         "1", "100%", "10%"]
            })
            st.dataframe(df_y_axis, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Información adicional
        st.markdown("<div class='section-header'>📚 Documentación Técnica</div>", unsafe_allow_html=True)
        
        col_doc1, col_doc2 = st.columns(2)
        
        with col_doc1:
            st.markdown("""
            **📖 Fórmulas utilizadas:**
            - **Presión (P):** `P = (1.42 × 42 × t² × L) / (1000 × V)`
              - `t` = espesor (mm)
              - `L` = longitud (mm)
              - `V` = apertura de matriz (mm)
            
            **📏 Relación H/V:**
            - Longitud mínima de pliegue basada en apertura V
            """)
        
        with col_doc2:
            st.markdown("""
            **⚙️ Modelo de Machine Learning:**
            - **Algoritmo:** Random Forest Regressor
            - **Estimadores:** 1000 árboles
            - **Validación:** 80/20 train-test split
            - **Métricas:** MAE y R²
            
            **🔄 Actualización continua:**
            - Los datos se sincronizan con GitHub
            - El modelo se reentrena con cada predicción
            """)


if __name__ == "__main__":
    main()
