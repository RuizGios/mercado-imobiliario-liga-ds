# -*- coding: utf-8 -*-
"""
Evolução e Previsão de Preços Imobiliários
Liga de Data Science - UNICAMP
Streamlit App
"""

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import io
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Mercado Imobiliário · Liga DS UNICAMP",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg:        #0d0f14;
    --surface:   #161a24;
    --card:      #1e2330;
    --border:    #2a3044;
    --accent:    #7b5ea7;
    --accent2:   #c084fc;
    --gold:      #f4c542;
    --text:      #e8eaf0;
    --muted:     #8892a4;
    --green:     #4ade80;
    --red:       #f87171;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

h1, h2, h3, h4 {
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    color: var(--text);
}

/* Cards */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}

/* KPI tiles */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.kpi {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.kpi-label {
    font-size: 12px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'Space Mono', monospace;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--accent2);
    font-family: 'Space Mono', monospace;
}
.kpi-sub {
    font-size: 12px;
    color: var(--muted);
    margin-top: 4px;
}

/* Badge */
.badge {
    display: inline-block;
    background: rgba(123,94,167,0.2);
    border: 1px solid var(--accent);
    color: var(--accent2);
    font-size: 11px;
    font-family: 'Space Mono', monospace;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 6px;
}

/* Section title */
.section-title {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 32px 0 20px;
}
.section-title h2 {
    margin: 0;
    font-size: 22px;
}
.section-line {
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* Table */
.dataframe {
    background: var(--card) !important;
    color: var(--text) !important;
}

/* Metric overrides */
[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
}
[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-size: 12px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent2) !important;
    font-family: 'Space Mono', monospace;
}

/* Tab styling */
.stTabs [role="tablist"] {
    border-bottom: 1px solid var(--border);
}
.stTabs [role="tab"] {
    color: var(--muted);
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    color: var(--accent2) !important;
    border-bottom: 2px solid var(--accent2) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #9b59b6);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    padding: 10px 24px;
    transition: opacity 0.2s;
}
.stButton > button:hover {
    opacity: 0.85;
    color: white;
}

/* Selectbox / slider */
.stSelectbox > div, .stSlider {
    background: var(--card);
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--card) !important;
    border-radius: 8px;
    color: var(--text) !important;
}

/* Divider */
hr { border-color: var(--border); }

/* Sidebar nav */
.nav-item {
    padding: 10px 16px;
    border-radius: 8px;
    margin: 4px 0;
    cursor: pointer;
    transition: background 0.15s;
    font-weight: 500;
}
.nav-item:hover { background: rgba(123,94,167,0.15); }
.nav-item.active {
    background: rgba(123,94,167,0.25);
    border-left: 3px solid var(--accent2);
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#1e2330",
    "axes.facecolor":    "#1e2330",
    "axes.edgecolor":    "#2a3044",
    "axes.labelcolor":   "#e8eaf0",
    "axes.titlecolor":   "#e8eaf0",
    "xtick.color":       "#8892a4",
    "ytick.color":       "#8892a4",
    "text.color":        "#e8eaf0",
    "grid.color":        "#2a3044",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "legend.facecolor":  "#1e2330",
    "legend.edgecolor":  "#2a3044",
    "figure.dpi":        130,
})

PALETTE = ["#c084fc", "#7b5ea7", "#4ade80", "#f4c542", "#f87171", "#60a5fa"]


# ──────────────────────────────────────────────
# DATA LOADING & CACHING
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_clean_data():
    """Download and clean the real estate dataset."""
    try:
        import kagglehub
        path = kagglehub.dataset_download("omniamahmoudsaeed/real-estate-sales-2001-2022")
        csv_path = f"{path}/Real_Estate_Sales_2001-2022_GL.csv"
    except Exception:
        st.error("❌ Não foi possível baixar o dataset do Kaggle. Configure as credenciais do Kaggle.")
        st.stop()

    df = pd.read_csv(csv_path)
    df_clean = df.copy()

    # Duplicates
    df_clean = df_clean.drop_duplicates()

    # Date parsing
    if "Date Recorded" in df_clean.columns:
        df_clean["Date Recorded"] = pd.to_datetime(df_clean["Date Recorded"], errors="coerce")
        df_clean["Month"] = df_clean["Date Recorded"].dt.month
        df_clean["Year"]  = df_clean["Date Recorded"].dt.year

    # Remove zero / negative Sale Amount & Assessed Value
    df_clean = df_clean[df_clean["Sale Amount"] > 0]
    df_clean = df_clean[df_clean["Assessed Value"] > 0]

    # Remove top 1% outliers on Sale Amount
    p99 = df_clean["Sale Amount"].quantile(0.99)
    df_clean = df_clean[df_clean["Sale Amount"] <= p99]

    # Fill nulls in Residential Type
    df_clean["Residential Type"] = df_clean["Residential Type"].fillna("Desconhecido")

    # Standardise column names
    df_clean.columns = (
        df_clean.columns.str.strip().str.lower().str.replace(" ", "_")
    )

    # Encode residential_type for scatter/viz
    res_map = {
        "Single Family": 0, "Condo": 1, "Two Family": 2,
        "Three Family": 3, "Four Family": 4, "Desconhecido": 5,
    }
    df_clean["residential_type_enc"] = (
        df_clean["residential_type"].map(res_map).fillna(5).astype(int)
    )

    return df_clean


@st.cache_data(show_spinner=False)
def prepare_model_data(_df):
    """Encode + scale features and split train/test."""
    COLS_TO_DROP = [
        "serial_number", "date_recorded", "address",
        "assessor_remarks", "opm_remarks", "location",
        "residential_type_enc",
    ]
    existing_drop = [c for c in COLS_TO_DROP if c in _df.columns]
    df_model = _df.drop(columns=existing_drop)

    TARGET = "sale_amount"
    cat_cols = df_model.select_dtypes(include=["object"]).columns.tolist()

    df_enc = df_model.copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_enc[cat_cols] = enc.fit_transform(df_enc[cat_cols].astype(str))

    y = df_enc[TARGET].copy()
    X = df_enc.drop(columns=TARGET)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    return X_train, X_test, y_train, y_test, X.columns.tolist()


@st.cache_data(show_spinner=False)
def train_models(_X_train, _X_test, _y_train, _y_test):
    """Train all four models and return results."""
    models = {
        "Decision Tree":   DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest":   RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42),
        "Linear Regression": LinearRegression(),
        "Ridge":           Ridge(alpha=1.0),
    }
    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(_X_train, _y_train)
        preds = model.predict(_X_test)
        results[name] = {
            "MAE":  mean_absolute_error(_y_test, preds),
            "RMSE": mean_squared_error(_y_test, preds) ** 0.5,
            "R²":   r2_score(_y_test, preds),
        }
        trained[name] = model
    return results, trained


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 24px'>
        <div style='font-family:Space Mono;font-size:11px;color:#8892a4;letter-spacing:2px;'>LIGA DE DATA SCIENCE</div>
        <div style='font-size:20px;font-weight:700;margin:6px 0;color:#e8eaf0;'>🏠 Mercado<br>Imobiliário</div>
        <div style='font-family:Space Mono;font-size:10px;color:#7b5ea7;'>UNICAMP · 2026</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navegação",
        ["📋 Visão Geral", "🔍 Exploração (EDA)", "📈 Evolução Temporal", "🤖 Modelos ML", "🎯 Simulador"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px;color:#8892a4;line-height:1.8'>
        <b style='color:#c084fc'>Dataset</b><br>
        Real Estate Sales<br>Connecticut, EUA<br>
        2001 – 2022<br><br>
        <b style='color:#c084fc'>Modelos</b><br>
        Decision Tree<br>
        Random Forest<br>
        Linear Regression<br>
        Ridge Regression
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
with st.spinner("⏳ Carregando e limpando dataset..."):
    df = load_and_clean_data()

# ──────────────────────────────────────────────
# PAGE: VISÃO GERAL
# ──────────────────────────────────────────────
if page == "📋 Visão Geral":
    st.markdown("""
    <div style='padding:32px 0 8px'>
        <div style='font-family:Space Mono;font-size:11px;color:#7b5ea7;letter-spacing:3px;'>LIGA DE DATA SCIENCE · UNICAMP</div>
        <h1 style='font-size:36px;font-weight:800;margin:8px 0;'>Evolução e Previsão de<br>Preços Imobiliários</h1>
        <p style='color:#8892a4;font-size:16px;max-width:700px;'>
            Análise de mais de 20 anos de dados de vendas imobiliárias em Connecticut, EUA,
            com modelos preditivos para estimar o valor de venda de imóveis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    total_vendas  = len(df)
    media_preco   = df["sale_amount"].mean()
    mediana_preco = df["sale_amount"].median()
    n_cidades     = df["town"].nunique() if "town" in df.columns else "—"
    anos_span     = f"{int(df['year'].min())}–{int(df['year'].max())}" if "year" in df.columns else "—"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Registros", f"{total_vendas:,}".replace(",", "."))
    with col2:
        st.metric("Preço Médio de Venda", f"${media_preco:,.0f}")
    with col3:
        st.metric("Mediana do Preço", f"${mediana_preco:,.0f}")
    with col4:
        st.metric("Cidades / Período", f"{n_cidades} cidades · {anos_span}")

    st.markdown("---")

    # Sprint cards
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="card">
            <div style='color:#c084fc;font-family:Space Mono;font-size:11px;letter-spacing:2px;'>SPRINT 1</div>
            <h3 style='margin:8px 0 12px;font-size:18px;'>🔍 EDA</h3>
            <ul style='color:#8892a4;font-size:14px;line-height:2;padding-left:16px;'>
                <li>Entendimento da base</li>
                <li>Limpeza e padronização</li>
                <li>Correlações iniciais</li>
                <li>Hipóteses sobre preços</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="card">
            <div style='color:#4ade80;font-family:Space Mono;font-size:11px;letter-spacing:2px;'>SPRINT 2</div>
            <h3 style='margin:8px 0 12px;font-size:18px;'>🤖 Modelagem</h3>
            <ul style='color:#8892a4;font-size:14px;line-height:2;padding-left:16px;'>
                <li>Encoding & Normalização</li>
                <li>Treino/Teste (75/25)</li>
                <li>4 modelos de regressão</li>
                <li>Avaliação MAE e RMSE</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="card">
            <div style='color:#f4c542;font-family:Space Mono;font-size:11px;letter-spacing:2px;'>SPRINT 3</div>
            <h3 style='margin:8px 0 12px;font-size:18px;'>🎯 Refinamento</h3>
            <ul style='color:#8892a4;font-size:14px;line-height:2;padding-left:16px;'>
                <li>Ajuste de modelos</li>
                <li>Interpretação de resultados</li>
                <li>Fatores mais relevantes</li>
                <li>Apresentação final</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick data preview
    with st.expander("👁️ Visualizar amostra dos dados limpos"):
        st.dataframe(df.head(50), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]:,} linhas × {df.shape[1]} colunas")

    st.markdown("""
    <div class="card" style='margin-top:8px'>
        <div style='font-family:Space Mono;font-size:11px;color:#8892a4;letter-spacing:2px;margin-bottom:12px;'>SOBRE O DATASET</div>
        <p style='color:#8892a4;font-size:14px;line-height:1.7;margin:0'>
            O dataset <b style='color:#e8eaf0'>Real Estate Sales 2001–2022</b> contém registros de vendas imobiliárias
            do estado de Connecticut, EUA. Cada linha representa uma transação com informações como
            cidade, tipo de propriedade, valor avaliado, valor de venda e data do registro.
            Com mais de 20 anos de histórico, é possível identificar tendências de mercado,
            analisar o impacto de eventos econômicos nos preços e construir modelos preditivos robustos.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PAGE: EDA
# ──────────────────────────────────────────────
elif page == "🔍 Exploração (EDA)":
    st.markdown("<h1 style='font-size:32px;font-weight:800;'>🔍 Análise Exploratória</h1>", unsafe_allow_html=True)
    st.caption("Entendendo a distribuição, correlações e padrões dos dados")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribuição de Preços",
        "Correlações",
        "Tipos de Propriedade",
        "Top Cidades",
    ])

    # ── Tab 1: Distribuição ──
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(9, 4))
            sns.histplot(
                data=df, x="sale_amount", bins=60, kde=True,
                color="#c084fc", ax=ax, line_kws={"linewidth": 2}
            )
            ax.set_title("Distribuição dos Valores de Venda", fontsize=14, fontweight="bold", pad=15)
            ax.set_xlabel("Valor de Venda (USD)")
            ax.set_ylabel("Frequência")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}K"))
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col2:
            st.markdown("<div class='card' style='margin-top:32px'>", unsafe_allow_html=True)
            st.markdown("**📊 Estatísticas Descritivas**")
            stats = df["sale_amount"].describe()
            for label, key in [("Mínimo", "min"), ("25%", "25%"), ("Mediana", "50%"),
                                ("75%", "75%"), ("Máximo", "max"), ("Média", "mean"), ("Std Dev", "std")]:
                val = stats[key] if key in stats else df["sale_amount"].mean()
                st.markdown(f"<div style='display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #2a3044'><span style='color:#8892a4;font-size:13px'>{label}</span><span style='font-family:Space Mono;font-size:13px;color:#c084fc'>${val:,.0f}</span></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Sales Ratio
        if "sales_ratio" in df.columns:
            fig2, ax2 = plt.subplots(figsize=(9, 3.5))
            sns.histplot(data=df, x="sales_ratio", bins=60, kde=True, color="#4ade80", ax=ax2)
            ax2.set_title("Distribuição da Razão de Vendas (Sales Ratio)", fontsize=13, fontweight="bold", pad=12)
            ax2.set_xlabel("Razão de Vendas")
            ax2.set_ylabel("Frequência")
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

    # ── Tab 2: Correlações ──
    with tab2:
        numerical_df = df.select_dtypes(include=["number"])
        df_corr = numerical_df.corr()

        fig, ax = plt.subplots(figsize=(11, 8))
        mask = np.zeros_like(df_corr, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True

        cmap = sns.diverging_palette(260, 20, as_cmap=True)
        sns.heatmap(
            df_corr, annot=True, fmt=".2f", cmap=cmap,
            linewidths=0.5, linecolor="#0d0f14",
            ax=ax, cbar_kws={"shrink": 0.7},
            annot_kws={"size": 8},
        )
        ax.set_title("Matriz de Correlação entre Variáveis Numéricas", fontsize=14, fontweight="bold", pad=15)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.info("💡 **Dica de leitura:** Valores próximos de +1 (roxo escuro) indicam correlação positiva forte. Próximos de -1 (vermelho escuro) indicam correlação negativa. Próximos de 0 indicam fraca relação linear.")

    # ── Tab 3: Tipos de Propriedade ──
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            prop_counts = df["property_type"].value_counts().head(8)
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.barh(prop_counts.index, prop_counts.values, color=PALETTE[:len(prop_counts)])
            ax.set_title("Tipos de Propriedade (Top 8)", fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel("Quantidade de Vendas")
            for bar in bars:
                ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
                        f"{bar.get_width():,.0f}", va="center", fontsize=8, color="#8892a4")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col2:
            if "residential_type" in df.columns:
                res_order = ["Single Family", "Condo", "Two Family", "Three Family", "Four Family", "Desconhecido"]
                res_price = (
                    df[df["residential_type"].isin(res_order)]
                    .groupby("residential_type")["sale_amount"].median()
                    .reindex([r for r in res_order if r in df["residential_type"].unique()])
                )
                fig, ax = plt.subplots(figsize=(7, 4))
                bars = ax.barh(res_price.index, res_price.values, color=PALETTE[:len(res_price)])
                ax.set_title("Mediana do Preço por Tipo Residencial", fontsize=13, fontweight="bold", pad=12)
                ax.set_xlabel("Mediana do Valor de Venda (USD)")
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

    # ── Tab 4: Top Cidades ──
    with tab4:
        n_top = st.slider("Número de cidades", 5, 20, 10)
        sales_by_town = df.groupby("town")["sale_amount"].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, max(4, n_top * 0.45)))
        top = sales_by_town.head(n_top)
        colors = [PALETTE[0] if i == 0 else PALETTE[1] if i < 3 else "#3d4663" for i in range(len(top))]
        bars = ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
        ax.set_title(f"Top {n_top} Cidades por Média de Preço de Vendas", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Média do Valor de Venda (USD)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}K"))
        for bar in bars:
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                    f"${bar.get_width():,.0f}", va="center", fontsize=8, color="#8892a4")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ──────────────────────────────────────────────
# PAGE: EVOLUÇÃO TEMPORAL
# ──────────────────────────────────────────────
elif page == "📈 Evolução Temporal":
    st.markdown("<h1 style='font-size:32px;font-weight:800;'>📈 Evolução Temporal</h1>", unsafe_allow_html=True)
    st.caption("Como o mercado imobiliário mudou ao longo dos anos")
    st.markdown("---")

    # Yearly average price
    yearly_avg = df.groupby("year")["sale_amount"].mean().reset_index()
    yearly_count = df.groupby("year")["sale_amount"].count().reset_index()
    yearly_median = df.groupby("year")["sale_amount"].median().reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(yearly_avg["year"], yearly_avg["sale_amount"],
                color="#c084fc", linewidth=2.5, marker="o", markersize=5)
        ax.fill_between(yearly_avg["year"], yearly_avg["sale_amount"],
                        alpha=0.15, color="#c084fc")
        ax.set_title("Média do Valor de Venda por Ano", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Ano")
        ax.set_ylabel("Média (USD)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(yearly_count["year"], yearly_count["sale_amount"],
                color="#4ade80", linewidth=2.5, marker="o", markersize=5)
        ax.fill_between(yearly_count["year"], yearly_count["sale_amount"],
                        alpha=0.15, color="#4ade80")
        ax.set_title("Volume de Vendas por Ano", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Ano")
        ax.set_ylabel("Nº de Vendas")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Single family evolution
    if "residential_type" in df.columns:
        st.markdown("---")
        st.markdown("**🏡 Imóveis Unifamiliares (Single Family)**")

        df_sf = df[df["residential_type"] == "Single Family"].copy()
        sf_yearly = (
            df_sf.groupby(df_sf["date_recorded"].dt.to_period("Y"))["sale_amount"]
            .agg(["count", "mean", "median"])
            .reset_index()
        )
        sf_yearly["date_recorded"] = sf_yearly["date_recorded"].dt.start_time

        col3, col4 = st.columns(2)
        with col3:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(sf_yearly["date_recorded"], sf_yearly["count"],
                    color="#f4c542", linewidth=2.5, marker="o", markersize=6)
            ax.fill_between(sf_yearly["date_recorded"], sf_yearly["count"],
                            alpha=0.15, color="#f4c542")
            ax.set_title("Vendas Anuais — Single Family", fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel("Ano")
            ax.set_ylabel("Nº de Vendas")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col4:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(sf_yearly["date_recorded"], sf_yearly["mean"],
                    color="#f87171", linewidth=2.5, marker="o", markersize=6, label="Média")
            ax.plot(sf_yearly["date_recorded"], sf_yearly["median"],
                    color="#60a5fa", linewidth=2, linestyle="--", marker="s", markersize=4, label="Mediana")
            ax.fill_between(sf_yearly["date_recorded"], sf_yearly["mean"],
                            alpha=0.1, color="#f87171")
            ax.set_title("Preço Single Family ao Longo dos Anos", fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel("Ano")
            ax.set_ylabel("Valor (USD)")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # Total revenue per year
    st.markdown("---")
    st.markdown("**💰 Volume Total Transacionado por Ano**")
    yearly_total = df.groupby("year")["sale_amount"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(yearly_total["year"], yearly_total["sale_amount"],
                  color=[PALETTE[0] if i % 2 == 0 else PALETTE[1] for i in range(len(yearly_total))],
                  edgecolor="#0d0f14", linewidth=0.5)
    ax.set_title("Volume Total de Vendas por Ano (USD)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Ano")
    ax.set_ylabel("Total (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e9:.1f}B"))
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ──────────────────────────────────────────────
# PAGE: MODELOS ML
# ──────────────────────────────────────────────
elif page == "🤖 Modelos ML":
    st.markdown("<h1 style='font-size:32px;font-weight:800;'>🤖 Modelos de Machine Learning</h1>", unsafe_allow_html=True)
    st.caption("Comparação entre 4 algoritmos de regressão para prever o valor de venda")
    st.markdown("---")

    with st.spinner("Treinando modelos... isso pode demorar alguns segundos ⏳"):
        X_train, X_test, y_train, y_test, feature_names = prepare_model_data(df)
        results, trained_models = train_models(X_train, X_test, y_train, y_test)

    df_res = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Modelo"})

    st.success(f"✅ Modelos treinados com {len(X_train):,} amostras de treino e {len(X_test):,} de teste.")

    # KPI metrics per model
    cols = st.columns(4)
    colors_badge = {"Decision Tree": "#c084fc", "Random Forest": "#4ade80",
                    "Linear Regression": "#f4c542", "Ridge": "#f87171"}
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            c = colors_badge.get(name, "#c084fc")
            st.markdown(f"""
            <div class="card" style="border-top:3px solid {c}">
                <div style='font-size:11px;color:{c};font-family:Space Mono;letter-spacing:1px;margin-bottom:8px'>{name.upper()}</div>
                <div style='margin-bottom:6px'>
                    <span style='color:#8892a4;font-size:12px'>MAE</span><br>
                    <span style='font-family:Space Mono;font-size:18px;color:{c}'>${res["MAE"]:,.0f}</span>
                </div>
                <div style='margin-bottom:6px'>
                    <span style='color:#8892a4;font-size:12px'>RMSE</span><br>
                    <span style='font-family:Space Mono;font-size:16px;color:#e8eaf0'>${res["RMSE"]:,.0f}</span>
                </div>
                <div>
                    <span style='color:#8892a4;font-size:12px'>R²</span><br>
                    <span style='font-family:Space Mono;font-size:16px;color:#e8eaf0'>{res["R²"]:.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        modelo_names = df_res["Modelo"]
        maes = df_res["MAE"]
        bar_colors = list(colors_badge.values())
        bars = ax.barh(modelo_names, maes, color=bar_colors, edgecolor="#0d0f14", height=0.5)
        ax.set_title("MAE por Modelo\n(Menor é melhor)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("MAE — Erro Absoluto Médio (USD)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
        for bar in bars:
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                    f"${bar.get_width():,.0f}", va="center", fontsize=8, color="#8892a4")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        rmses = df_res["RMSE"]
        bars = ax.barh(modelo_names, rmses, color=bar_colors, edgecolor="#0d0f14", height=0.5)
        ax.set_title("RMSE por Modelo\n(Menor é melhor)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("RMSE — Raiz do Erro Quadrático Médio (USD)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
        for bar in bars:
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                    f"${bar.get_width():,.0f}", va="center", fontsize=8, color="#8892a4")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Best model highlight
    best_model_name = min(results, key=lambda k: results[k]["MAE"])
    best = results[best_model_name]
    st.markdown(f"""
    <div class="card" style="border-left:4px solid #4ade80;margin-top:8px">
        <div style='font-family:Space Mono;font-size:11px;color:#4ade80;letter-spacing:2px;margin-bottom:8px'>🏆 MELHOR MODELO (menor MAE)</div>
        <b style='font-size:20px'>{best_model_name}</b>
        <div style='color:#8892a4;margin-top:8px;font-size:14px;line-height:1.8'>
            MAE: <b style='color:#e8eaf0'>${best["MAE"]:,.0f}</b> &nbsp;·&nbsp;
            RMSE: <b style='color:#e8eaf0'>${best["RMSE"]:,.0f}</b> &nbsp;·&nbsp;
            R²: <b style='color:#e8eaf0'>{best["R²"]:.4f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics explanation
    with st.expander("📚 Entendendo as métricas de avaliação"):
        st.markdown("""
        **MAE — Mean Absolute Error (Erro Absoluto Médio)**
        > A média dos erros absolutos entre o valor real e o predito. Em dólares: em média, o modelo erra por esse valor. Quanto menor, melhor.

        **RMSE — Root Mean Squared Error (Raiz do Erro Quadrático Médio)**
        > Similar ao MAE, mas penaliza erros grandes com mais força (por elevar ao quadrado antes de tirar a média). Quanto menor, melhor.

        **R² — Coeficiente de Determinação**
        > Indica quanto da variação do preço é explicada pelo modelo. Varia de 0 a 1: quanto mais perto de 1, melhor o ajuste.
        """)

    # Feature importance (Random Forest)
    if "Random Forest" in trained_models:
        st.markdown("---")
        st.markdown("**🌲 Importância das Features — Random Forest**")
        rf = trained_models["Random Forest"]
        importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.barh(importances.index[::-1], importances.values[::-1],
                       color=[PALETTE[0] if i < 3 else "#3d4663" for i in range(len(importances)-1, -1, -1)])
        ax.set_title("Top 10 Features mais Importantes (Random Forest)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Importância")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ──────────────────────────────────────────────
# PAGE: SIMULADOR
# ──────────────────────────────────────────────
elif page == "🎯 Simulador":
    st.markdown("<h1 style='font-size:32px;font-weight:800;'>🎯 Simulador de Preços</h1>", unsafe_allow_html=True)
    st.caption("Explore como diferentes variáveis afetam o valor estimado de um imóvel")
    st.markdown("---")

    # Train needed
    with st.spinner("Preparando modelos..."):
        X_train, X_test, y_train, y_test, feature_names = prepare_model_data(df)
        results, trained_models = train_models(X_train, X_test, y_train, y_test)

    st.markdown("""
    <div class="card">
        <div style='font-family:Space Mono;font-size:11px;color:#8892a4;letter-spacing:2px;margin-bottom:12px;'>COMO FUNCIONA</div>
        <p style='color:#8892a4;font-size:14px;margin:0;line-height:1.7'>
            Ajuste os parâmetros abaixo para simular diferentes cenários de imóveis.
            O valor estimado é calculado em tempo real usando o Random Forest,
            o modelo com menor erro no conjunto de teste.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("**⚙️ Parâmetros do Imóvel**")

        list_year = st.slider("Ano de Listagem", 2001, 2022, 2015)
        assessed_value = st.number_input("Valor Avaliado (USD)", min_value=10_000, max_value=5_000_000,
                                          value=300_000, step=10_000, format="%d")
        sales_ratio = st.slider("Sales Ratio (Razão de Venda)", 0.5, 3.0, 1.0, step=0.05)
        month = st.slider("Mês da Venda", 1, 12, 6)
        year = st.slider("Ano da Venda", 2001, 2022, 2015)

        res_type_label = st.selectbox(
            "Tipo Residencial",
            ["Single Family", "Condo", "Two Family", "Three Family", "Four Family", "Desconhecido"]
        )
        res_type_enc = {"Single Family": 0, "Condo": 1, "Two Family": 2,
                        "Three Family": 3, "Four Family": 4, "Desconhecido": 5}[res_type_label]

        prop_type_label = st.selectbox(
            "Tipo de Propriedade",
            ["Residential", "Commercial", "Industrial", "Vacant Land", "Apartments"]
        )

        town_options = sorted(df["town"].dropna().unique().tolist()) if "town" in df.columns else ["Greenwich"]
        town = st.selectbox("Cidade (Town)", town_options, index=0)

    with col_result:
        st.markdown("**📊 Estimativa de Valor**")

        # Build input row aligned with training features
        COLS_TO_DROP_SIM = ["serial_number", "date_recorded", "address",
                            "assessor_remarks", "opm_remarks", "location", "residential_type_enc"]

        # Get a sample row and replace values
        try:
            df_sample = df.drop(columns=[c for c in COLS_TO_DROP_SIM if c in df.columns], errors="ignore")
            df_sample = df_sample.drop(columns=["sale_amount"], errors="ignore")

            cat_cols_sim = df_sample.select_dtypes(include=["object"]).columns.tolist()
            enc_sim = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            df_sample_enc = df_sample.copy()
            df_sample_enc[cat_cols_sim] = enc_sim.fit_transform(df_sample_enc[cat_cols_sim].astype(str))

            input_row = df_sample_enc.iloc[[0]].copy()

            # Map fields to column names
            field_map = {
                "assessed_value": assessed_value,
                "sales_ratio": sales_ratio,
                "list_year": list_year,
                "month": month,
                "year": year,
            }
            for col, val in field_map.items():
                if col in input_row.columns:
                    input_row[col] = val

            # Encode town and property_type
            if "town" in cat_cols_sim and "town" in input_row.columns:
                town_enc_val = enc_sim.transform([[town] + [""] * (len(cat_cols_sim) - 1)])[0][cat_cols_sim.index("town")]
                input_row["town"] = town_enc_val

            if "property_type" in cat_cols_sim and "property_type" in input_row.columns:
                prop_idx = cat_cols_sim.index("property_type")
                row_for_enc = [""] * len(cat_cols_sim)
                row_for_enc[prop_idx] = prop_type_label
                prop_enc_val = enc_sim.transform([row_for_enc])[0][prop_idx]
                input_row["property_type"] = prop_enc_val

            if "residential_type" in cat_cols_sim and "residential_type" in input_row.columns:
                res_idx = cat_cols_sim.index("residential_type")
                row_for_enc2 = [""] * len(cat_cols_sim)
                row_for_enc2[res_idx] = res_type_label
                res_enc_val = enc_sim.transform([row_for_enc2])[0][res_idx]
                input_row["residential_type"] = res_enc_val

            # Predict
            rf_model = trained_models["Random Forest"]
            pred_value = rf_model.predict(input_row[feature_names])[0]

            # Display
            st.markdown(f"""
            <div class="card" style="border-top:3px solid #c084fc;text-align:center;padding:32px">
                <div style='font-family:Space Mono;font-size:11px;color:#8892a4;letter-spacing:2px;margin-bottom:12px'>VALOR ESTIMADO</div>
                <div style='font-size:44px;font-weight:800;font-family:Space Mono;color:#c084fc;'>${pred_value:,.0f}</div>
                <div style='color:#8892a4;margin-top:8px;font-size:13px'>via Random Forest Regressor</div>
            </div>
            """, unsafe_allow_html=True)

            # Range estimate
            rf_mae = results["Random Forest"]["MAE"]
            low = max(0, pred_value - rf_mae)
            high = pred_value + rf_mae

            st.markdown(f"""
            <div class="card" style="margin-top:12px">
                <div style='font-family:Space Mono;font-size:11px;color:#8892a4;letter-spacing:2px;margin-bottom:12px'>INTERVALO ESTIMADO (±MAE)</div>
                <div style='display:flex;justify-content:space-between;'>
                    <div>
                        <div style='color:#8892a4;font-size:12px'>Mínimo</div>
                        <div style='font-family:Space Mono;color:#4ade80;font-size:18px'>${low:,.0f}</div>
                    </div>
                    <div style='color:#3d4663;font-size:24px;align-self:center'>↔</div>
                    <div style='text-align:right'>
                        <div style='color:#8892a4;font-size:12px'>Máximo</div>
                        <div style='font-family:Space Mono;color:#f87171;font-size:18px'>${high:,.0f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Model comparison for this input
            st.markdown("**Estimativas por modelo:**")
            for name, model in trained_models.items():
                pred = model.predict(input_row[feature_names])[0]
                c = list(colors_badge.values())[list(trained_models.keys()).index(name)]
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;padding:8px 12px;background:#1e2330;border-radius:8px;margin-bottom:6px;border-left:3px solid {c}'>
                    <span style='font-size:13px;color:#8892a4'>{name}</span>
                    <span style='font-family:Space Mono;font-size:14px;color:{c}'>${pred:,.0f}</span>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"Não foi possível gerar a estimativa: {e}")
            st.info("Navegue até a página de Modelos ML primeiro para treinar os modelos.")


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:16px 0;color:#3d4663;font-family:Space Mono;font-size:11px;'>
    Liga de Data Science · UNICAMP · Faculdade de Tecnologia
    &nbsp;|&nbsp; Dataset: Real Estate Sales 2001–2022 · Kaggle
</div>
""", unsafe_allow_html=True)
