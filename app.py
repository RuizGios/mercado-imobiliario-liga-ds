# -*- coding: utf-8 -*-
"""
Evolução e Previsão de Preços Imobiliários
Liga de Data Science - UNICAMP
Streamlit App — versão corrigida e formal
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
from sklearn.preprocessing import OrdinalEncoder
import io, os

warnings.filterwarnings("ignore")
np.random.seed(42)

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Mercado Imobiliário · Liga DS UNICAMP",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS BASEADO NO MANUAL DA MARCA LIGA DS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&display=swap');

:root {
    --bg:      #000000; /* Preto (Cor Secundária) */
    --surface: #0C6980; /* Azul Escuro (Cor Secundária) */
    --card:    #094D5E; /* Tom levemente mais escuro do Azul para profundidade nos cards */
    --border:  #3891AC; /* Azul de Apoio para bordas */
    --accent:  #FED531; /* Amarelo (Cor Principal) */
    --accent2: #FFE684; /* Amarelo Claro (Cor de Apoio) */
    --text:    #FFFFFF; /* Branco (Cor de Apoio) */
    --muted:   #A0D0DF; /* Azul muito clarinho para textos secundários */
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
h1,h2,h3,h4 { font-family:'Inter',sans-serif; font-weight:600; color:var(--accent); letter-spacing: -0.02em; }
.card {
    background:var(--card); border:1px solid var(--border);
    border-radius:8px; padding:24px; margin-bottom:16px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
}
[data-testid="metric-container"] {
    background:var(--card); border:1px solid var(--border);
    border-radius:8px; padding:16px;
}
[data-testid="metric-container"] label { color:var(--text)!important; font-size:13px!important; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:var(--accent)!important; font-family:'Fira Code',monospace; font-weight: 500;
}
.stTabs [role="tablist"] { border-bottom:1px solid var(--border); }
.stTabs [role="tab"] { color:var(--muted); font-family:'Inter',sans-serif; font-weight:500; }
.stTabs [aria-selected="true"] { color:var(--accent)!important; border-bottom:2px solid var(--accent)!important; }
.stButton>button {
    background:var(--accent);
    color:#000000; border:none; border-radius:6px;
    font-family:'Inter',sans-serif; font-weight:700;
    padding:10px 24px; transition:opacity .2s;
}
.stButton>button:hover { opacity:.85; color:#000000; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# MATPLOTLIB BRAND THEME
# ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#000000", "axes.facecolor": "#000000",
    "axes.edgecolor": "#3891AC", "axes.labelcolor": "#FFFFFF",
    "axes.titlecolor": "#FED531", "xtick.color": "#A0D0DF",
    "ytick.color": "#A0D0DF", "text.color": "#FFFFFF",
    "grid.color": "#3891AC", "grid.linestyle": "--", "grid.alpha": 0.4,
    "legend.facecolor": "#094D5E", "legend.edgecolor": "#3891AC",
    "figure.dpi": 130,
})
# Substituindo a paleta antiga pelas cores da Liga DS
PALETTE = ["#FED531", "#FFE684", "#3891AC", "#0C6980", "#FFFFFF", "#A0D0DF"]
COLORS_BADGE = {
    "Decision Tree": "#3891AC", "Random Forest": "#FED531",
    "Linear Regression": "#FFFFFF", "Ridge": "#FFE684"
}

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if "df_loaded" not in st.session_state:
    st.session_state["df_loaded"] = None

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 24px'>
        <div style='font-family:Fira Code;font-size:11px;color:#94a3b8;letter-spacing:1px;text-transform:uppercase;'>Liga de Data Science</div>
        <div style='font-size:20px;font-weight:600;margin:12px 0;color:#f8fafc;'>Mercado<br>Imobiliário</div>
        <div style='font-family:Fira Code;font-size:10px;color:#3b82f6;'>UNICAMP · 2025</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navegação",
        ["Visão Geral","Exploração (EDA)","Evolução Temporal","Modelos de ML","Simulador"],
        label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px;color:#94a3b8;line-height:1.8'>
        <b style='color:#3b82f6'>Dataset</b><br>
        Real Estate Sales<br>Connecticut, EUA<br>2001 – 2022<br><br>
        <b style='color:#3b82f6'>Modelos</b><br>
        Decision Tree · Random Forest<br>Linear Regression · Ridge
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
COLS_TO_DROP = ["serial_number","date_recorded","address",
                "assessor_remarks","opm_remarks","location","residential_type_enc"]

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_FILENAME = "Real_Estate_Sales_2001-2022_GL.zip"
CSV_PATH     = os.path.join(_SCRIPT_DIR, CSV_FILENAME)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    if "Date Recorded" in df.columns:
        df["Date Recorded"] = pd.to_datetime(df["Date Recorded"], errors="coerce")
        df["Month"] = df["Date Recorded"].dt.month
        df["Year"]  = df["Date Recorded"].dt.year
    if "Sale Amount"    in df.columns: df = df[df["Sale Amount"] > 0]
    if "Assessed Value" in df.columns: df = df[df["Assessed Value"] > 0]
    if "Sale Amount"    in df.columns:
        p99 = df["Sale Amount"].quantile(0.99)
        df  = df[df["Sale Amount"] <= p99]
    if "Residential Type" in df.columns:
        df["Residential Type"] = df["Residential Type"].fillna("Desconhecido")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ","_")
    res_map = {"Single Family":0,"Condo":1,"Two Family":2,
               "Three Family":3,"Four Family":4,"Desconhecido":5}
    if "residential_type" in df.columns:
        df["residential_type_enc"] = df["residential_type"].map(res_map).fillna(5).astype(int)
    return df

@st.cache_data(show_spinner=False)
def load_from_local(path: str) -> pd.DataFrame:
    return clean_df(pd.read_csv(path))

@st.cache_data(show_spinner=False)
def load_from_kaggle() -> pd.DataFrame:
    import kagglehub
    path = kagglehub.dataset_download("omniamahmoudsaeed/real-estate-sales-2001-2022")
    return clean_df(pd.read_csv(f"{path}/{CSV_FILENAME}"))

df = st.session_state["df_loaded"]

if df is None:
    _candidates = [CSV_PATH, CSV_FILENAME,
                   os.path.join("/workspaces/mercado-imobiliario-liga-ds", CSV_FILENAME),
                   os.path.join(os.getcwd(), CSV_FILENAME)]
    _found = next((p for p in _candidates if os.path.exists(p)), None)
    if _found:
        with st.spinner(f"Carregando base de dados..."):
            df = load_from_local(_found)
        st.session_state["df_loaded"] = df

if df is None:
    try:
        with st.spinner("Conectando ao Kaggle..."):
            df = load_from_kaggle()
        st.session_state["df_loaded"] = df
    except Exception:
        df = None

if df is None:
    st.markdown("""
    <div style='max-width:660px;margin:60px auto 0;text-align:center'>
        <h2 style='font-size:26px;font-weight:600;margin-bottom:8px'>Dataset não encontrado</h2>
        <p style='color:#94a3b8;font-size:15px;line-height:1.7;margin-bottom:24px'>
            O arquivo é grande demais para upload pelo navegador.<br>
            Por favor, insira-o diretamente no diretório do projeto.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ──────────────────────────────────────────────
# PREPARE MODEL DATA
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def prepare_and_train(_df):
    drop_existing = [c for c in COLS_TO_DROP if c in _df.columns]
    df_model = _df.drop(columns=drop_existing)

    TARGET    = "sale_amount"
    cat_cols  = df_model.select_dtypes(include=["object"]).columns.tolist()

    df_enc = df_model.copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_enc[cat_cols] = enc.fit_transform(df_enc[cat_cols].astype(str))

    y = df_enc[TARGET].copy()
    X = df_enc.drop(columns=TARGET)

    num_cols_x = X.select_dtypes(include=["number"]).columns
    X[num_cols_x] = X[num_cols_x].fillna(X[num_cols_x].median())
    X = X.fillna(-1)
    y = y.fillna(y.median())

    feat_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    models = {
        "Decision Tree":     DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42),
        "Linear Regression": LinearRegression(),
        "Ridge":             Ridge(alpha=1.0),
    }
    results, trained = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "MAE":  mean_absolute_error(y_test, preds),
            "RMSE": mean_squared_error(y_test, preds) ** 0.5,
            "R²":   r2_score(y_test, preds),
        }
        trained[name] = model

    return X_train, X_test, y_train, y_test, feat_names, enc, cat_cols, results, trained


# ══════════════════════════════════════════════
# PÁGINAS
# ══════════════════════════════════════════════

# ──────────────────────────────────────────────
# VISÃO GERAL
# ──────────────────────────────────────────────
if page == "Visão Geral":
    st.markdown("""
    <div style='padding:32px 0 8px'>
        <div style='font-family:Fira Code;font-size:11px;color:#3b82f6;letter-spacing:2px;text-transform:uppercase;'>Liga de Data Science · UNICAMP</div>
        <h1 style='font-size:36px;font-weight:600;margin:8px 0;'>Evolução e Previsão de<br>Preços Imobiliários</h1>
        <p style='color:#94a3b8;font-size:16px;max-width:700px;'>
            Análise de dados históricos referentes a vendas imobiliárias no estado de Connecticut (EUA),
            incluindo a implementação de modelos preditivos para estimativa de valores de venda.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Total de Registros", f"{len(df):,}".replace(",","."))
    with c2: st.metric("Preço Médio", f"${df['sale_amount'].mean():,.0f}")
    with c3: st.metric("Mediana do Preço", f"${df['sale_amount'].median():,.0f}")
    with c4:
        span = f"{int(df['year'].min())}–{int(df['year'].max())}" if "year" in df.columns else "—"
        st.metric("Período", span)

    st.markdown("---")
    ca,cb,cc = st.columns(3)
    cards = [
        ("#d2e1f9","FASE 1","Exploração de Dados",["Entendimento da base","Limpeza e padronização","Análise de correlações","Validação de hipóteses"]),
        ("#CBFFF9","FASE 2","Modelagem e Previsão",["Codificação de variáveis","Divisão de dados (75/25)","Treinamento de regressores","Avaliação de métricas"]),
        ("#fff1c8","FASE 3","Refinamento",["Ajuste de hiperparâmetros","Interpretação de resultados","Extração de fatores relevantes","Apresentação de negócios"]),
    ]
    for col,(color,sprint,title,items) in zip([ca,cb,cc],cards):
        with col:
            li = "".join(f"<li>{i}</li>" for i in items)
            st.markdown(f"""
            <div class="card">
                <div style='color:{color};font-family:Fira Code;font-size:11px;letter-spacing:1px;text-transform:uppercase;'>{sprint}</div>
                <h3 style='margin:8px 0 12px;font-size:16px;'>{title}</h3>
                <ul style='color:#94a3b8;font-size:14px;line-height:2;padding-left:16px;'>{li}</ul>
            </div>""", unsafe_allow_html=True)

    with st.expander("Visualizar amostra dos dados"):
        st.dataframe(df.head(50), use_container_width=True)
        st.caption(f"Dimensões: {df.shape[0]:,} linhas × {df.shape[1]} colunas")

    if st.button("Recarregar dataset"):
        st.session_state["df_loaded"] = None
        st.cache_data.clear()
        st.rerun()

# ──────────────────────────────────────────────
# EDA
# ──────────────────────────────────────────────
elif page == "Exploração (EDA)":
    st.markdown("<h1 style='font-size:28px;font-weight:600;'>Análise Exploratória</h1>", unsafe_allow_html=True)
    st.caption("Distribuição, correlações e padrões estruturais do conjunto de dados")
    st.markdown("---")

    tab1,tab2,tab3,tab4 = st.tabs(["Distribuição de Preços","Correlações","Tipos de Propriedade","Cidades em Destaque"])

    with tab1:
        col1,col2 = st.columns([2,1])
        with col1:
            fig,ax = plt.subplots(figsize=(9,4))
            sns.histplot(data=df, x="sale_amount", bins=60, kde=True, color="#3b82f6", ax=ax, line_kws={"linewidth":2})
            ax.set_title("Distribuição dos Valores de Venda", fontsize=12, fontweight="600", pad=15)
            ax.set_xlabel("Valor de Venda (USD)")
            ax.set_ylabel("Frequência")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x/1e6:.1f}M" if x>=1e6 else f"${x/1e3:.0f}K"))
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        with col2:
            st.markdown("<div class='card' style='margin-top:32px'>", unsafe_allow_html=True)
            st.markdown("**Estatísticas Descritivas**")
            stats = df["sale_amount"].describe()
            for label,key in [("Mínimo","min"),("25%","25%"),("Mediana","50%"),
                              ("75%","75%"),("Máximo","max"),("Média","mean"),("Desvio Padrão","std")]:
                val = stats[key] if key in stats else df["sale_amount"].mean()
                st.markdown(f"<div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #334155'><span style='color:#94a3b8;font-size:13px'>{label}</span><span style='font-family:Fira Code;font-size:13px;color:#f8fafc'>${val:,.0f}</span></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if "sales_ratio" in df.columns:
            fig2,ax2 = plt.subplots(figsize=(9,3.5))
            sns.histplot(data=df, x="sales_ratio", bins=60, kde=True, color="#6366f1", ax=ax2)
            ax2.set_title("Distribuição da Razão de Vendas (Sales Ratio)", fontsize=12, fontweight="600", pad=12)
            ax2.set_xlabel("Razão de Vendas"); ax2.set_ylabel("Frequência")
            fig2.tight_layout(); st.pyplot(fig2, use_container_width=True); plt.close()

    with tab2:
        numerical_df = df.select_dtypes(include=["number"])
        df_corr = numerical_df.corr()
        fig,ax = plt.subplots(figsize=(11,8))
        sns.heatmap(df_corr, annot=True, fmt=".2f",
                    cmap=sns.diverging_palette(220, 20, as_cmap=True),
                    linewidths=.5, linecolor="#0f172a", ax=ax,
                    cbar_kws={"shrink":.7}, annot_kws={"size":9})
        ax.set_title("Matriz de Correlação Linear", fontsize=12, fontweight="600", pad=15)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        st.info("Nota: Valores próximos de +1 indicam correlação positiva forte. Valores próximos de 0 indicam fraca relação linear.")

    with tab3:
        col1,col2 = st.columns(2)
        with col1:
            prop_counts = df["property_type"].value_counts().head(8)
            fig,ax = plt.subplots(figsize=(7,4))
            ax.barh(prop_counts.index, prop_counts.values, color=PALETTE[:len(prop_counts)])
            ax.set_title("Tipos de Propriedade (Top 8)", fontsize=12, fontweight="600", pad=12)
            ax.set_xlabel("Quantidade de Transações")
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        with col2:
            if "residential_type" in df.columns:
                res_order = ["Single Family","Condo","Two Family","Three Family","Four Family","Desconhecido"]
                res_price = (df[df["residential_type"].isin(res_order)]
                             .groupby("residential_type")["sale_amount"].median()
                             .reindex([r for r in res_order if r in df["residential_type"].unique()]))
                fig,ax = plt.subplots(figsize=(7,4))
                ax.barh(res_price.index, res_price.values, color=PALETTE[:len(res_price)])
                ax.set_title("Mediana do Preço por Tipo Residencial", fontsize=12, fontweight="600", pad=12)
                ax.set_xlabel("Mediana (USD)")
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x/1e3:.0f}K"))
                fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with tab4:
        n_top = st.slider("Quantidade de cidades para visualizar", 5, 20, 10)
        sales_by_town = df.groupby("town")["sale_amount"].mean().sort_values(ascending=False)
        top = sales_by_town.head(n_top)
        colors_bar = [PALETTE[0] if i==0 else PALETTE[1] if i<3 else "#334155" for i in range(len(top))]
        fig,ax = plt.subplots(figsize=(10, max(4, n_top*0.45)))
        ax.barh(top.index[::-1], top.values[::-1], color=colors_bar[::-1])
        ax.set_title(f"Top {n_top} Cidades — Média de Preço de Venda", fontsize=12, fontweight="600", pad=15)
        ax.set_xlabel("Média (USD)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x/1e6:.1f}M" if x>=1e6 else f"${x/1e3:.0f}K"))
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

# ──────────────────────────────────────────────
# EVOLUÇÃO TEMPORAL
# ──────────────────────────────────────────────
elif page == "Evolução Temporal":
    st.markdown("<h1 style='font-size:28px;font-weight:600;'>Evolução Temporal</h1>", unsafe_allow_html=True)
    st.caption("Métricas do mercado imobiliário analisadas por período")
    st.markdown("---")

    yearly_avg   = df.groupby("year")["sale_amount"].mean().reset_index()
    yearly_count = df.groupby("year")["sale_amount"].count().reset_index()

    col1,col2 = st.columns(2)
    with col1:
        fig,ax = plt.subplots(figsize=(8,4))
        ax.plot(yearly_avg["year"], yearly_avg["sale_amount"], color="#3b82f6", linewidth=2.5, marker="o", markersize=5)
        ax.fill_between(yearly_avg["year"], yearly_avg["sale_amount"], alpha=.15, color="#3b82f6")
        ax.set_title("Média do Valor de Venda Anual", fontsize=12, fontweight="600", pad=12)
        ax.set_xlabel("Ano"); ax.set_ylabel("Média (USD)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x/1e3:.0f}K"))
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
    with col2:
        fig,ax = plt.subplots(figsize=(8,4))
        ax.plot(yearly_count["year"], yearly_count["sale_amount"], color="#10b981", linewidth=2.5, marker="o", markersize=5)
        ax.fill_between(yearly_count["year"], yearly_count["sale_amount"], alpha=.15, color="#10b981")
        ax.set_title("Volume de Transações por Ano", fontsize=12, fontweight="600", pad=12)
        ax.set_xlabel("Ano"); ax.set_ylabel("Nº de Vendas")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    if "residential_type" in df.columns and "date_recorded" in df.columns:
        st.markdown("---")
        st.markdown("**Imóveis Unifamiliares (Single Family)**")
        df_sf = df[df["residential_type"]=="Single Family"].copy()
        sf_yearly = (df_sf.groupby(df_sf["date_recorded"].dt.to_period("Y"))["sale_amount"]
                     .agg(["count","mean","median"]).reset_index())
        sf_yearly["date_recorded"] = sf_yearly["date_recorded"].dt.start_time

        col3,col4 = st.columns(2)
        with col3:
            fig,ax = plt.subplots(figsize=(8,4))
            ax.plot(sf_yearly["date_recorded"], sf_yearly["count"], color="#eab308", linewidth=2.5, marker="o", markersize=6)
            ax.fill_between(sf_yearly["date_recorded"], sf_yearly["count"], alpha=.15, color="#eab308")
            ax.set_title("Vendas Anuais — Single Family", fontsize=12, fontweight="600", pad=12)
            ax.set_xlabel("Ano"); ax.set_ylabel("Nº de Vendas")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        with col4:
            fig,ax = plt.subplots(figsize=(8,4))
            ax.plot(sf_yearly["date_recorded"], sf_yearly["mean"], color="#ef4444", linewidth=2.5, marker="o", markersize=6, label="Média")
            ax.plot(sf_yearly["date_recorded"], sf_yearly["median"], color="#8b5cf6", linewidth=2, linestyle="--", marker="s", markersize=4, label="Mediana")
            ax.fill_between(sf_yearly["date_recorded"], sf_yearly["mean"], alpha=.1, color="#ef4444")
            ax.set_title("Evolução de Preço — Single Family", fontsize=12, fontweight="600", pad=12)
            ax.set_xlabel("Ano"); ax.set_ylabel("Valor (USD)")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x/1e3:.0f}K"))
            ax.legend()
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")
    st.markdown("**Volume Total Transacionado por Ano**")
    yearly_total = df.groupby("year")["sale_amount"].sum().reset_index()
    fig,ax = plt.subplots(figsize=(12,4))
    ax.bar(yearly_total["year"], yearly_total["sale_amount"],
           color=[PALETTE[0] if i%2==0 else PALETTE[1] for i in range(len(yearly_total))],
           edgecolor="#0f172a", linewidth=.5)
    ax.set_title("Volume Total de Vendas por Ano (USD)", fontsize=12, fontweight="600", pad=15)
    ax.set_xlabel("Ano"); ax.set_ylabel("Total (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x/1e9:.1f}B"))
    fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

# ──────────────────────────────────────────────
# MODELOS ML
# ──────────────────────────────────────────────
elif page == "Modelos de ML":
    st.markdown("<h1 style='font-size:28px;font-weight:600;'>Modelos de Machine Learning</h1>", unsafe_allow_html=True)
    st.caption("Comparação de desempenho entre algoritmos de regressão para estimativa de preços")
    st.markdown("---")

    with st.spinner("Processando treinamento dos modelos..."):
        X_train,X_test,y_train,y_test,feat_names,enc,cat_cols,results,trained = prepare_and_train(df)

    st.success(f"Modelos treinados com sucesso: {len(X_train):,} amostras em treinamento e {len(X_test):,} em teste.")

    cols_kpi = st.columns(4)
    for i,(name,res) in enumerate(results.items()):
        c = list(COLORS_BADGE.values())[i]
        with cols_kpi[i]:
            st.markdown(f"""
            <div class="card" style="border-top:3px solid {c}">
                <div style='font-size:11px;color:{c};font-family:Fira Code;letter-spacing:1px;margin-bottom:8px;text-transform:uppercase;'>{name}</div>
                <div style='margin-bottom:6px'><span style='color:#94a3b8;font-size:12px'>MAE</span><br>
                    <span style='font-family:Fira Code;font-size:18px;color:{c}'>${res["MAE"]:,.0f}</span></div>
                <div style='margin-bottom:6px'><span style='color:#94a3b8;font-size:12px'>RMSE</span><br>
                    <span style='font-family:Fira Code;font-size:16px;color:#f8fafc'>${res["RMSE"]:,.0f}</span></div>
                <div><span style='color:#94a3b8;font-size:12px'>R²</span><br>
                    <span style='font-family:Fira Code;font-size:16px;color:#f8fafc'>{res["R²"]:.4f}</span></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    df_res = pd.DataFrame(results).T.reset_index().rename(columns={"index":"Modelo"})
    bar_colors = list(COLORS_BADGE.values())

    col1,col2 = st.columns(2)
    with col1:
        fig,ax = plt.subplots(figsize=(7,4))
        ax.barh(df_res["Modelo"], df_res["MAE"], color=bar_colors, edgecolor="#0f172a", height=.5)
        ax.set_title("MAE por Modelo (Menor é melhor)", fontsize=12, fontweight="600", pad=12)
        ax.set_xlabel("Erro Absoluto Médio (USD)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x/1e3:.0f}K"))
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
    with col2:
        fig,ax = plt.subplots(figsize=(7,4))
        ax.barh(df_res["Modelo"], df_res["RMSE"], color=bar_colors, edgecolor="#0f172a", height=.5)
        ax.set_title("RMSE por Modelo (Menor é melhor)", fontsize=12, fontweight="600", pad=12)
        ax.set_xlabel("RMSE (USD)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x/1e3:.0f}K"))
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    best_name = min(results, key=lambda k: results[k]["MAE"])
    best = results[best_name]
    st.markdown(f"""
    <div class="card" style="border-left:4px solid #10b981">
        <div style='font-family:Fira Code;font-size:11px;color:#10b981;letter-spacing:1px;margin-bottom:8px'>MELHOR MODELO IDENTIFICADO</div>
        <b style='font-size:18px'>{best_name}</b>
        <div style='color:#94a3b8;margin-top:8px;font-size:14px;line-height:1.8'>
            MAE: <b style='color:#f8fafc'>${best["MAE"]:,.0f}</b> &nbsp;·&nbsp;
            RMSE: <b style='color:#f8fafc'>${best["RMSE"]:,.0f}</b> &nbsp;·&nbsp;
            R²: <b style='color:#f8fafc'>{best["R²"]:.4f}</b>
        </div>
    </div>""", unsafe_allow_html=True)

    with st.expander("Entendendo as métricas de avaliação"):
        st.markdown("""
        **MAE (Erro Absoluto Médio):** Representa a média das diferenças absolutas entre as previsões e os valores reais. Expressa o erro médio esperado na mesma unidade do dado (dólares).

        **RMSE (Raiz do Erro Quadrático Médio):** Penaliza mais fortemente grandes desvios entre a previsão e o valor real, sendo útil para identificar a estabilidade do modelo contra *outliers*.

        **R² (Coeficiente de Determinação):** Indica a proporção da variância na variável dependente que é previsível a partir das variáveis independentes (1.0 = precisão perfeita, 0.0 = sem capacidade preditiva).
        """)

    if "Random Forest" in trained:
        st.markdown("---")
        st.markdown("**Importância das Features — Random Forest**")
        importances = pd.Series(trained["Random Forest"].feature_importances_, index=feat_names).sort_values(ascending=False).head(10)
        fig,ax = plt.subplots(figsize=(10,4))
        ax.barh(importances.index[::-1], importances.values[::-1],
                color=[PALETTE[0] if i<3 else "#334155" for i in range(len(importances)-1,-1,-1)])
        ax.set_title("Top 10 Features de Maior Relevância", fontsize=12, fontweight="600", pad=12)
        ax.set_xlabel("Grau de Importância Relativa")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

# ──────────────────────────────────────────────
# SIMULADOR
# ──────────────────────────────────────────────
elif page == "Simulador":
    st.markdown("<h1 style='font-size:28px;font-weight:600;'>Simulador de Preços</h1>", unsafe_allow_html=True)
    st.caption("Projeção do valor imobiliário com base na parametrização de variáveis do modelo")
    st.markdown("---")

    with st.spinner("Inicializando motores de predição..."):
        X_train,X_test,y_train,y_test,feat_names,enc,cat_cols,results,trained = prepare_and_train(df)

    st.markdown("""
    <div class="card">
        <div style='font-family:Fira Code;font-size:11px;color:#94a3b8;letter-spacing:1px;margin-bottom:8px;text-transform:uppercase;'>INSTRUÇÕES DO MÓDULO</div>
        <p style='color:#94a3b8;font-size:14px;margin:0;line-height:1.7'>
            Ajuste as características do imóvel. O cálculo estimado é gerado em tempo real,
            utilizando as rotinas e encoders configurados no treinamento de dados.
        </p>
    </div>""", unsafe_allow_html=True)

    col_form, col_res = st.columns([1,1])

    with col_form:
        st.markdown("**Parâmetros do Imóvel**")
        list_year      = st.slider("Ano de Listagem", 2001, 2022, 2015)
        assessed_value = st.number_input("Valor Avaliado (USD)", 10_000, 5_000_000, 300_000, 10_000, format="%d")
        sales_ratio    = st.slider("Sales Ratio", 0.5, 3.0, 1.0, 0.05)
        month          = st.slider("Mês da Venda", 1, 12, 6)
        year_val       = st.slider("Ano da Venda", 2001, 2022, 2015)

        res_opts = ["Single Family","Condo","Two Family","Three Family","Four Family","Desconhecido"]
        res_label = st.selectbox("Tipo Residencial", res_opts)

        prop_opts = sorted(df["property_type"].dropna().unique().tolist()) if "property_type" in df.columns else ["Residential"]
        prop_label = st.selectbox("Tipo de Propriedade", prop_opts)

        town_opts = sorted(df["town"].dropna().unique().tolist()) if "town" in df.columns else ["Greenwich"]
        town_label = st.selectbox("Cidade", town_opts)

    with col_res:
        st.markdown("**Estimativa de Valor**")

        try:
            drop_existing = [c for c in COLS_TO_DROP if c in df.columns]
            df_base = df.drop(columns=drop_existing + ["sale_amount"], errors="ignore")
            input_row = df_base.iloc[[0]].copy()

            num_map = {
                "assessed_value": assessed_value,
                "sales_ratio":    sales_ratio,
                "list_year":      list_year,
                "month":          month,
                "year":           year_val,
            }
            for col_n, val in num_map.items():
                if col_n in input_row.columns:
                    input_row[col_n] = float(val)

            cat_map = {
                "town":             town_label,
                "property_type":    prop_label,
                "residential_type": res_label,
            }
            for col_c, val in cat_map.items():
                if col_c in input_row.columns:
                    input_row[col_c] = val

            input_row[cat_cols] = enc.transform(input_row[cat_cols].astype(str))
            X_input = input_row[feat_names]

            # CORREÇÃO DO BUG AQUI: Preenchimento de qualquer valor vazio (NaN) residual
            # usando -1 (seguindo o padrão estabelecido no tratamento base da função prepare_and_train)
            X_input = X_input.fillna(-1)

            preds_all = {name: model.predict(X_input)[0] for name, model in trained.items()}
            pred_rf   = preds_all["Random Forest"]
            mae_rf    = results["Random Forest"]["MAE"]

            st.markdown(f"""
            <div class="card" style="border-top:3px solid #3b82f6;text-align:center;padding:32px">
                <div style='font-family:Fira Code;font-size:11px;color:#94a3b8;letter-spacing:1px;margin-bottom:12px;text-transform:uppercase'>VALOR ESTIMADO · RANDOM FOREST</div>
                <div style='font-size:40px;font-weight:600;font-family:Fira Code;color:#3b82f6;'>${pred_rf:,.0f}</div>
                <div style='color:#94a3b8;margin-top:8px;font-size:13px'>Margem de Erro (MAE): ${max(0,pred_rf-mae_rf):,.0f} — ${pred_rf+mae_rf:,.0f}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("**Estimativas por modelo algorítmico:**")
            for name, pred in preds_all.items():
                c = COLORS_BADGE[name]
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;padding:10px 14px;
                     background:var(--surface);border-radius:6px;margin-bottom:6px;border-left:3px solid {c}'>
                    <span style='font-size:13px;color:#94a3b8;font-weight:500;'>{name}</span>
                    <span style='font-family:Fira Code;font-size:14px;color:{c};font-weight:600;'>${pred:,.0f}</span>
                </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Erro na execução da estimativa: {e}")
            st.info("Verifique a integridade do modelo ou tente recarregar os dados na aba Visão Geral.")

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:16px 0;color:#64748b;font-family:Fira Code;font-size:11px;'>
    Liga de Data Science · UNICAMP &nbsp;|&nbsp; Fonte de Dados: Real Estate Sales 2001–2022
</div>""", unsafe_allow_html=True)