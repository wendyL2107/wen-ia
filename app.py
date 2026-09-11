# -*- coding: utf-8 -*-
"""
Wen IA — WORKBENCH DE EXPLORACIÓN NUMÉRICA (Streamlit Edition)
Suite Científica Multivariable para Datos Experimentales (.xlsx, .xls, .ods)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import streamlit as st

# Módulos de Aprendizaje Automático, Clustering y Reducción Dimensional
try:
    from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
    from scipy.spatial.distance import pdist
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_DISPONIBLE = True
except ImportError:
    SKLEARN_DISPONIBLE = False

try:
    import shap
    SHAP_DISPONIBLE = True
except ImportError:
    SHAP_DISPONIBLE = False

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILO DE Wen IA
# ==============================================================================
st.set_page_config(
    page_title="Wen IA — Workbench Científico",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLOR_BASE_BG = "#f8f9fa"
COLOR_CARD_BG = "#ffffff"
COLOR_BORDER = "#e5d8dc"
COLOR_ROSE_DEEP = "#b84d66"
COLOR_ROSE_ACCENT = "#c25d72"
COLOR_ROSE_LIGHT = "#fbebed"
COLOR_TEXT_MAIN = "#1f1e24"
COLOR_TEXT_MUTED = "#5e5863"

# Inyección de estilos CSS personalizados para mantener la estética
st.markdown(f"""
    <style>
    .main {{
        background-color: {COLOR_BASE_BG};
    }}
    .badge-wen {{
        background-color: {COLOR_ROSE_DEEP};
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }}
    .metric-box {{
        background-color: {COLOR_CARD_BG};
        border: 1px solid {COLOR_BORDER};
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }}
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# MOTOR MATEMÁTICO (PRESERVADO INTACTO)
# ==============================================================================
class MotorAnalitico:
    @staticmethod
    def calcular_correlaciones(df, target):
        corr_todas = df.corr()
        corr_target = corr_todas[target].drop(target)
        corr_ordenada = corr_target.abs().sort_values(ascending=False)
        return corr_todas, corr_target, corr_ordenada

    @staticmethod
    def entrenar_random_forest(df, target):
        X = df.drop(columns=[target])
        y = df[target]

        if SKLEARN_DISPONIBLE:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importancias = pd.Series(
                rf.feature_importances_, index=X.columns
            ).sort_values(ascending=False)
            return rf, importancias, "Random Forest Regressor"
        else:
            correlaciones = df.corr()[target].drop(target)
            r2 = correlaciones**2
            importancias = (r2 / r2.sum()).sort_values(ascending=False)
            return None, importancias, "Varianza Explicada Normalizada (R²)"

    @staticmethod
    def calcular_pca_biplot(df):
        if not SKLEARN_DISPONIBLE:
            return None, None, None, None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df.values)

        pca = PCA(n_components=2)
        scores = pca.fit_transform(X_scaled)
        loadings = pca.components_
        var_explicada = pca.explained_variance_ratio_ * 100
        return scores, loadings, var_explicada, list(df.columns)

    @staticmethod
    def calcular_clustering_jerarquico(df, metodo_enlace="ward", k_clusters=3):
        if not SKLEARN_DISPONIBLE:
            return None, None, None, None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df.values)

        Z = linkage(X_scaled, method=metodo_enlace)
        c, _ = cophenet(Z, pdist(X_scaled))

        n_clusters = min(k_clusters, len(df))
        asignaciones = fcluster(Z, t=n_clusters, criterion="maxclust")
        return Z, c, asignaciones, n_clusters

    @staticmethod
    def calcular_topsis(df, pesos=None, criterios_beneficio=None):
        X = df.values.astype(float)
        n_muestras, n_criterios = X.shape

        norm = np.linalg.norm(X, axis=0)
        norm[norm == 0] = 1.0
        X_norm = X / norm

        if pesos is None:
            w = np.ones(n_criterios) / n_criterios
        else:
            w = np.array(pesos) / np.sum(pesos)
        X_pond = X_norm * w

        if criterios_beneficio is None:
            criterios_beneficio = [True] * n_criterios

        a_pos = np.zeros(n_criterios)
        a_neg = np.zeros(n_criterios)

        for j in range(n_criterios):
            if criterios_beneficio[j]:
                a_pos[j] = np.max(X_pond[:, j])
                a_neg[j] = np.min(X_pond[:, j])
            else:
                a_pos[j] = np.min(X_pond[:, j])
                a_neg[j] = np.max(X_pond[:, j])

        d_pos = np.sqrt(np.sum((X_pond - a_pos) ** 2, axis=1))
        d_neg = np.sqrt(np.sum((X_pond - a_neg) ** 2, axis=1))

        denom = d_pos + d_neg
        denom[denom == 0] = 1.0
        puntajes = d_neg / denom
        return puntajes, a_pos, a_neg

    @staticmethod
    def calcular_shap_explicabilidad(df, target):
        if not SKLEARN_DISPONIBLE:
            return None, None, None

        X = df.drop(columns=[target])
        y = df[target]
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        if SHAP_DISPONIBLE:
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X)
        else:
            base_pred = rf.predict(X)
            shap_values = np.zeros(X.shape)
            for col_idx, col_name in enumerate(X.columns):
                X_mod = X.copy()
                X_mod[col_name] = X[col_name].mean()
                pred_mod = rf.predict(X_mod)
                shap_values[:, col_idx] = base_pred - pred_mod

        return shap_values, X, list(X.columns)

    @staticmethod
    def detectar_anomalias(df):
        if not SKLEARN_DISPONIBLE:
            return None, None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df.values)

        iso = IsolationForest(contamination=0.15, random_state=42)
        preds_iso = iso.fit_predict(X_scaled)

        cov = np.cov(X_scaled, rowvar=False)
        cov_inv = np.linalg.pinv(cov)
        diff = X_scaled - np.mean(X_scaled, axis=0)
        dist_mahal = np.sqrt(np.sum(np.dot(diff, cov_inv) * diff, axis=1))
        return preds_iso, dist_mahal


# ==============================================================================
# ENCABEZADO PRINCIPAL DE Wen IA
# ==============================================================================
col_badge, col_title = st.columns([1, 6])
with col_badge:
    st.markdown('<div class="badge-wen">Wen IA</div>', unsafe_allow_html=True)
with col_title:
    st.markdown("### **W**orkbench de **E**xploración **N**umérica e **IA**")
    st.caption("Plataforma Integrada de Análisis Multivariable, Optimización e Inteligencia Artificial")

st.divider()

# ==============================================================================
# BARRA LATERAL (SIDEBAR): CARGA Y PARÁMETROS
# ==============================================================================
st.sidebar.header("📁 Datos de Entrada")
archivo_cargado = st.sidebar.file_uploader(
    "Carga tu matriz experimental (.xlsx, .xls, .ods)", 
    type=["xlsx", "xls", "ods"]
)

# Modal explicativo con st.expander
with st.sidebar.expander("📋 Guía Metodológica"):
    st.write("""
    * **Fila 1:** Nombres de las variables cuantitativas.
    * **Fila 2+:** Datos numéricos sin valores faltantes.
    * **Columna 1 (opcional):** Etiquetas de muestras (ej. P1, P2) para rotular gráficos.
    """)

if archivo_cargado is not None:
    try:
        df_raw = pd.read_excel(archivo_cargado, header=0)
        cols_num = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        df_num = df_raw[cols_num].dropna()

        N, M = df_num.shape
        if M < 2 or N < 3:
            st.error("Error: Se requieren al menos 3 filas y 2 columnas numéricas.")
            st.stop()

        cols_txt = [c for c in df_raw.columns if c not in cols_num]
        etiquetas = df_raw[cols_txt[0]].astype(str).tolist() if cols_txt else [f"M_{i+1}" for i in range(N)]

        st.sidebar.success(f"Matriz cargada: N={N} muestras, M={M} variables")
        
        # Selector de Variable Objetivo
        target = st.sidebar.selectbox("1. Variable Objetivo (Target):", cols_num)

        # Selector de Gráfico
        opciones_graficos = [
            "🔥 1. Mapa de Calor (Heatmap de Pearson)",
            "🧭 2. PCA Biplot 2D (Muestras y Cargas)",
            "🌳 3. Clustering Jerárquico (Dendrograma de Ward)",
            "🏆 4. Optimización Multicriterio (Ranking TOPSIS)",
            "📊 5. Importancia de Variables (Random Forest)",
            "🐝 6. Explicabilidad IA (SHAP & Beeswarm)",
            "🛡️ 7. Control de Calidad y Detección de Anomalías"
        ]
        grafico_sel = st.sidebar.selectbox("2. Módulo Científico Activo:", opciones_graficos)

        # Entrenar modelo
        rf_model, importancias, metodo = MotorAnalitico.entrenar_random_forest(df_num, target)
        _, corr_target, corr_ord = MotorAnalitico.calcular_correlaciones(df_num, target)

        # ======================================================================
        # PANELES PRINCIPALES: CONSOLA Y VISOR GRÁFICO
        # ======================================================================
        col_consola, col_grafico = st.columns([1, 1.4])

        with col_consola:
            st.subheader("Consola de Diagnóstico")
            
            top_var = corr_ord.index[0]
            efecto = "incrementar" if corr_target[top_var] > 0 else "disminuir"

            st.info(f"**Variable dominante:** `{top_var}`\n\nIncrementar `{top_var}` tiende a **{efecto}** `{target}`.")

            tab1, tab2 = st.tabs(["Correlación (Pearson)", f"Influencia ({metodo})"])
            with tab1:
                df_corr_show = pd.DataFrame({
                    "Variable": corr_ord.index,
                    "r": [corr_target[v] for v in corr_ord.index],
                    "Fuerza": ["Alta" if abs(corr_target[v]) >= 0.7 else ("Media" if abs(corr_target[v]) >= 0.4 else "Baja") for v in corr_ord.index]
                })
                st.dataframe(df_corr_show, use_container_width=True, hide_index=True)

            with tab2:
                df_imp_show = pd.DataFrame({
                    "Variable": importancias.index,
                    "Peso Relativo (%)": (importancias.values * 100).round(2)
                })
                st.dataframe(df_imp_show, use_container_width=True, hide_index=True)

            # Simulador What-If integrado en el panel izquierdo
            with st.expander("🔬 Simulador Predictivo In Silico (What-If)", expanded=False):
                st.caption(f"Ajusta parámetros de entrada para estimar '{target}'")
                features = [c for c in df_num.columns if c != target]
                input_vals = {}
                for f in features:
                    f_min = float(df_num[f].min())
                    f_max = float(df_num[f].max())
                    f_mean = float(df_num[f].mean())
                    if f_min == f_max:
                        f_max += 1.0
                    input_vals[f] = st.slider(f, f_min, f_max, f_mean)

                if rf_model is not None:
                    df_pred = pd.DataFrame([input_vals])
                    pred_res = rf_model.predict(df_pred)[0]
                    st.metric("Predicción Estimada", f"{pred_res:.4f}")
                    st.caption(f"Rango experimental: [{df_num[target].min():.2f} a {df_num[target].max():.2f}]")

        with col_grafico:
            st.subheader("Visor Gráfico Científico")
            fig = Figure(figsize=(7.5, 6), dpi=120, facecolor=COLOR_CARD_BG)
            ax = fig.add_subplot(111)
            ax.set_facecolor(COLOR_CARD_BG)

            # 1. HEATMAP
            if "1. Mapa de Calor" in grafico_sel:
                corr = df_num.corr()
                n_v = len(corr.columns)
                cmap_custom = LinearSegmentedColormap.from_list("RoseHeatmap", ["#2A4365", "#FFFFFF", COLOR_ROSE_DEEP], N=256)
                cax = ax.imshow(corr.values, cmap=cmap_custom, vmin=-1.0, vmax=1.0, aspect="auto")
                fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xticks(range(n_v))
                ax.set_yticks(range(n_v))
                ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels(corr.columns, fontsize=8)
                if n_v <= 15:
                    for i in range(n_v):
                        for j in range(n_v):
                            val = corr.iloc[i, j]
                            col_t = COLOR_CARD_BG if abs(val) > 0.45 else COLOR_TEXT_MAIN
                            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", color=col_t, fontsize=7.5)
                ax.set_title("MATRIZ DE CORRELACIÓN DE PEARSON", fontweight="bold", color=COLOR_TEXT_MAIN)

            # 2. PCA BIPLOT
            elif "2. PCA Biplot" in grafico_sel:
                scores, loadings, var_exp, vars_cols = MotorAnalitico.calcular_pca_biplot(df_num)
                if scores is not None:
                    xs, ys = scores[:, 0], scores[:, 1]
                    ax.scatter(xs, ys, color=COLOR_ROSE_DEEP, edgecolor=COLOR_TEXT_MAIN, s=75, alpha=0.9, label="Muestras")
                    for i, txt in enumerate(etiquetas):
                        ax.annotate(txt, (xs[i], ys[i]), xytext=(5, 5), textcoords="offset points", fontsize=8, color=COLOR_TEXT_MAIN)
                    escala = min(np.abs(xs).max(), np.abs(ys).max()) * 0.95
                    for j, var_name in enumerate(vars_cols):
                        vx, vy = loadings[0, j] * escala, loadings[1, j] * escala
                        ax.annotate("", xy=(vx, vy), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=COLOR_ROSE_ACCENT, lw=1.5))
                        ax.text(vx * 1.1, vy * 1.1, var_name, color=COLOR_TEXT_MAIN, fontsize=8, fontweight="bold", ha="center")
                    ax.axhline(0, color=COLOR_BORDER, linestyle="--")
                    ax.axvline(0, color=COLOR_BORDER, linestyle="--")
                    ax.set_xlabel(f"PC1 ({var_exp[0]:.2f}%)")
                    ax.set_ylabel(f"PC2 ({var_exp[1]:.2f}%)")
                    ax.set_title(f"PCA BIPLOT 2D (Varianza: {np.sum(var_exp):.2f}%)", fontweight="bold")
                    ax.legend()

            # 3. CLUSTERING
            elif "3. Clustering" in grafico_sel:
                Z, c, _, _ = MotorAnalitico.calcular_clustering_jerarquico(df_num, metodo_enlace="ward")
                if Z is not None:
                    umbral = 0.70 * Z[-1, 2]
                    dendrogram(Z, labels=etiquetas, ax=ax, leaf_rotation=45, color_threshold=umbral, above_threshold_color=COLOR_TEXT_MUTED)
                    ax.axhline(umbral, color=COLOR_ROSE_DEEP, linestyle="--", label=f"Corte ({umbral:.2f})")
                    ax.set_title(f"DENDROGRAMA DE SIMILITUD (Ward) | Cofenético: {c:.3f}", fontweight="bold")
                    ax.set_ylabel("Distancia Euclidiana Estandarizada")
                    ax.legend()

            # 4. TOPSIS
            elif "4. Optimización" in grafico_sel:
                pts, _, _ = MotorAnalitico.calcular_topsis(df_num)
                rank_df = pd.DataFrame({"Muestra": etiquetas, "Puntaje": pts}).sort_values(by="Puntaje", ascending=False)
                y_pos = np.arange(len(rank_df))
                cmap_top = LinearSegmentedColormap.from_list("RoseTopsis", [COLOR_ROSE_LIGHT, COLOR_ROSE_ACCENT, COLOR_ROSE_DEEP], N=256)
                barras = ax.barh(y_pos, rank_df["Puntaje"], color=cmap_top(rank_df["Puntaje"] / rank_df["Puntaje"].max()), edgecolor=COLOR_TEXT_MAIN)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(rank_df["Muestra"], fontweight="bold")
                ax.invert_yaxis()
                for barra in barras:
                    w = barra.get_width()
                    ax.text(w + 0.01, barra.get_y() + barra.get_height() / 2, f"{w:.4f}", va="center", fontsize=8)
                ax.set_title("RANKING TOPSIS DE PROXIMIDAD AL IDEAL (A+)", fontweight="bold")

            # 5. IMPORTANCIA RF
            elif "5. Importancia" in grafico_sel:
                vars_imp = importancias.index[::-1]
                vals_imp = (importancias.values * 100)[::-1]
                y_pos = np.arange(len(vars_imp))
                cmap_imp = LinearSegmentedColormap.from_list("RoseImp", [COLOR_ROSE_LIGHT, COLOR_ROSE_ACCENT, COLOR_ROSE_DEEP], N=256)
                barras = ax.barh(y_pos, vals_imp, color=cmap_imp(vals_imp / vals_imp.max()), edgecolor=COLOR_TEXT_MAIN)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(vars_imp, fontweight="bold")
                for barra in barras:
                    w = barra.get_width()
                    ax.text(w + 0.5, barra.get_y() + barra.get_height() / 2, f"{w:.2f}%", va="center", fontsize=8)
                ax.set_title(f"IMPORTANCIA DE VARIABLES SOBRE '{target}'", fontweight="bold")

            # 6. SHAP BEESWARM
            elif "6. Explicabilidad" in grafico_sel:
                shap_vals, X_f, f_names = MotorAnalitico.calcular_shap_explicabilidad(df_num, target)
                if shap_vals is not None:
                    sorted_idx = np.argsort(np.mean(np.abs(shap_vals), axis=0))
                    y_pos = np.arange(len(sorted_idx))
                    cmap_shap = LinearSegmentedColormap.from_list("RoseShap", ["#2A4365", "#E28B9F", COLOR_ROSE_DEEP], N=256)
                    for i, idx in enumerate(sorted_idx):
                        vals = shap_vals[:, idx]
                        feat_vals = X_f.iloc[:, idx].values
                        norm_f = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-8)
                        jitter = np.random.normal(0, 0.05, size=len(vals))
                        ax.scatter(vals, y_pos[i] + jitter, c=cmap_shap(norm_f), s=50, edgecolor=COLOR_TEXT_MAIN, lw=0.4)
                    ax.axvline(0, color=COLOR_ROSE_ACCENT, linestyle="--")
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels([f_names[i] for i in sorted_idx], fontweight="bold")
                    ax.set_title(f"EXPLICABILIDAD SHAP BEESWARM ('{target}')", fontweight="bold")

            # 7. ANOMALÍAS
            elif "7. Control de Calidad" in grafico_sel:
                preds_iso, d_mahal = MotorAnalitico.detectar_anomalias(df_num)
                if preds_iso is not None:
                    x_pos = np.arange(len(etiquetas))
                    cols_anom = [COLOR_ROSE_DEEP if p == -1 else COLOR_ROSE_LIGHT for p in preds_iso]
                    ax.bar(x_pos, d_mahal, color=cols_anom, edgecolor=COLOR_TEXT_MAIN)
                    umbral_m = np.percentile(d_mahal, 85)
                    ax.axhline(umbral_m, color=COLOR_ROSE_ACCENT, linestyle="--", label=f"Alerta ({umbral_m:.2f})")
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(etiquetas, rotation=45, ha="right")
                    ax.set_ylabel("Distancia de Mahalanobis")
                    ax.set_title("CONTROL DE CALIDAD: DETECCIÓN DE ANOMALÍAS", fontweight="bold")
                    ax.legend()

            ax.grid(True, linestyle=":", alpha=0.6, color=COLOR_BORDER)
            fig.tight_layout()
            st.pyplot(fig)

    except Exception as err:
        st.error(f"Error procesando la matriz: {err}")
else:
    st.info("👈 Por favor, carga tu archivo de datos (.xlsx, .ods) desde la barra lateral izquierda para iniciar el análisis.")