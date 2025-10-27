# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# ============================
# Fonctions de fit
# ============================
def logistic_4PL(x, bottom, top, EC50, hill):
    return bottom + (top - bottom) / (1 + (EC50 / x) ** hill)

def logistic_3PL(x, top, EC50, hill):
    return top / (1 + (EC50 / x) ** hill)

def polynomial_2(x, a, b, c):
    return a * x**2 + b * x + c

# ============================
# Helpers cache
# ============================
@st.cache_data(show_spinner=False)
def precompute_stats(df_serializable_tuple):
    x_col, comps, records = df_serializable_tuple
    df_local = pd.DataFrame.from_records(records)
    stats = {}
    for comp in comps:
        tmp = df_local[[x_col, comp]].dropna()
        grp = tmp.groupby(x_col).agg(['mean','std'])[comp].reset_index()
        stats[comp] = grp
    return stats

@st.cache_data(show_spinner=False)
def compute_fit_and_smooth(x_fit_tuple, y_fit_tuple, model, force, smooth_n=200):
    x_fit = np.array(x_fit_tuple, dtype=float)
    y_fit = np.array(y_fit_tuple, dtype=float)
    if len(x_fit) < 2 or np.min(x_fit) <= 0:
        x_smooth = np.linspace(np.min(x_fit) if len(x_fit)>0 else 0.1, np.max(x_fit) if len(x_fit)>0 else 1.0, smooth_n)
    else:
        x_smooth = np.logspace(np.log10(np.min(x_fit)), np.log10(np.max(x_fit)), smooth_n)
    popt = None; ec50 = None
    y_smooth = np.full_like(x_smooth, np.nan, dtype=float)
    try:
        if model == "4PL":
            p0=[np.min(y_fit), np.max(y_fit), np.median(x_fit), 1.0]
            bounds=([np.min(y_fit)-10, np.min(y_fit), 1e-6, 0.0],
                    [np.max(y_fit)+10, np.max(y_fit)*2.0, np.max(x_fit)*10.0, 5.0])
            popt,_ = curve_fit(logistic_4PL, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=5000)
            y_smooth = logistic_4PL(x_smooth, *popt); ec50 = float(popt[2])
        elif model == "3PL":
            p0=[np.max(y_fit), np.median(x_fit), 1.0]
            popt,_ = curve_fit(logistic_3PL, x_fit, y_fit, p0=p0, maxfev=5000)
            y_smooth = logistic_3PL(x_smooth, *popt); ec50 = float(popt[1])
        else:
            popt,_ = curve_fit(polynomial_2, x_fit, y_fit, maxfev=5000)
            y_smooth = polynomial_2(x_smooth, *popt); ec50=None
    except:
        popt = None; ec50 = None

    if force == "0%":
        y_smooth = np.clip(y_smooth, 0, None)
    elif force == "100%":
        y_smooth = np.clip(y_smooth, None, 100)
    elif force == "0+100%":
        y_smooth = np.clip(y_smooth, 0, 100)

    return {"x_smooth": x_smooth, "y_smooth": y_smooth, "popt": tuple(popt) if popt is not None else None, "ec50": ec50}

# ============================
# Page config
# ============================
st.set_page_config(page_title="HTRF Dose–Response Interactive", layout="wide")
st.title("📈 HTRF Dose–Response — interactive")

# ============================
# Upload flexible CSV/TXT
# ============================
uploaded_file = st.sidebar.file_uploader("Importer CSV/TXT", type=["csv", "txt", "csv.gz"])
if uploaded_file is None:
    st.sidebar.info("Importe un fichier CSV ou TXT avec colonnes de concentration et composés.")
    st.stop()

# Try multiple separators
separators = [',', ';', '\t']
df = None
for sep in separators:
    try:
        df_try = pd.read_csv(uploaded_file, sep=sep, engine='python')
        numeric_cols = df_try.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df = df_try
            break
    except Exception:
        continue

if df is None:
    st.sidebar.error("Impossible de lire le fichier. Vérifie qu'il est bien un CSV ou TXT avec séparateur ',' ';' ou tabulation.")
    st.stop()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.sidebar.error("Aucune colonne numérique détectée.")
    st.stop()

default_x_col = "[Cpd] µM" if "[Cpd] µM" in numeric_cols else numeric_cols[0]
x_col = st.sidebar.selectbox("Colonne concentration (abscisse)", options=numeric_cols, index=numeric_cols.index(default_x_col))

compounds = [c for c in df.columns if c != x_col]
if not compounds:
    st.sidebar.error("Aucune colonne de composé détectée.")
    st.stop()

# ============================
# Sidebar — choix des courbes
# ============================
st.sidebar.header("⚙️ Choix des courbes")
selected = st.sidebar.multiselect("Choisir les composés :", options=compounds, default=[compounds[0]], key="selected_cols")

# ============================
# Sidebar — apparence
# ============================
st.sidebar.header("🎨 Apparence & légende")
chart_title = st.sidebar.text_input("Titre graphique", "HTRF Dose–Response")
x_axis_title = st.sidebar.text_input("Titre axe X", x_col)
y_axis_title = st.sidebar.text_input("Titre axe Y", "Signal / %")
global_point_size = st.sidebar.slider("Taille points", 4, 15, 8, key="global_point_size")
global_line_width = st.sidebar.slider("Épaisseur ligne par défaut", 1, 5, 2, key="global_line_width")
show_legend = st.sidebar.checkbox("Afficher légende", True)
show_points_legend = st.sidebar.checkbox("Afficher légende points", False)
show_global_ec50 = st.sidebar.checkbox("Afficher EC50 dans légendes", True)
show_fit_type_global = st.sidebar.checkbox("Afficher type de fit dans légende (global)", False)

st.sidebar.header("📐 Axes")
show_xaxis = st.sidebar.checkbox("Afficher axe X", True)
show_yaxis = st.sidebar.checkbox("Afficher axe Y", True)
axis_line_width = st.sidebar.slider("Épaisseur axes", 1, 3, 2)
manual_xlim = st.sidebar.checkbox("Définir X min/max", False)
if manual_xlim:
    x_min = st.sidebar.number_input("X min", float(df[x_col].min()))
    x_max = st.sidebar.number_input("X max", float(df[x_col].max()))
else:
    x_min = x_max = None
manual_ylim = st.sidebar.checkbox("Définir Y min/max", False)
if manual_ylim:
    y_min = st.sidebar.number_input("Y min", 0.0)
    y_max = st.sidebar.number_input("Y max", 100.0)
else:
    y_min = y_max = None

st.sidebar.header("💾 Export")

# ============================
# Precompute stats
# ============================
df_records = df.to_dict(orient="records")
stats_dict = precompute_stats((x_col, compounds, df_records))
palette = px.colors.qualitative.Plotly

# ============================
# Build figure
# ============================
fig = go.Figure()
curve_data = []

for i, comp in enumerate(selected):
    grp = stats_dict[comp]
    x_unique = grp[x_col].values
    y_mean = grp["mean"].values
    y_std = grp["std"].values

    # default values
    color = st.session_state.get(f"color_{comp}", palette[i % len(palette)])
    model = st.session_state.get(f"model_{comp}", "4PL")
    force_val = st.session_state.get(f"force_{comp}", "Aucune")
    width = st.session_state.get(f"width_{comp}", global_line_width)
    show_points = st.session_state.get(f"points_{comp}", True)
    legend_text = st.session_state.get(f"legend_text_{comp}", comp)

    # prepare fit
    mask_pos = x_unique > 0
    x_fit = x_unique[mask_pos]
    y_fit = y_mean[mask_pos]
    if len(x_fit) < 2:
        x_smooth = x_unique
        y_smooth = np.full_like(x_smooth, np.nan, dtype=float)
        popt = None; ec50 = None
    else:
        fit_result = compute_fit_and_smooth(tuple(x_fit.tolist()), tuple(y_fit.tolist()), model, force_val)
        x_smooth = fit_result["x_smooth"]
        y_smooth = fit_result["y_smooth"]
        popt = fit_result["popt"]
        ec50 = fit_result["ec50"]

    curve_info = {"comp": comp, "x": x_unique, "y": y_mean, "y_std": y_std,
                  "x_smooth": x_smooth, "y_smooth": y_smooth, "color": color,
                  "width": width, "model": model, "popt": popt, "ec50": ec50, "force": force_val}
    curve_data.append(curve_info)

    # points
    if show_points:
        fig.add_trace(go.Scatter(
            x=x_unique, y=y_mean, mode="markers",
            error_y=dict(type="data", array=y_std, visible=True, thickness=1, width=4),
            marker=dict(color=color, size=global_point_size),
            name=comp if show_points_legend else None,
            showlegend=show_points_legend
        ))

    # line
    label = legend_text
    if show_global_ec50 and ec50 is not None:
        label += f" EC50={ec50:.2f} µM"
    if show_fit_type_global:
        label += f" ({model})"

    fig.add_trace(go.Scatter(
        x=x_smooth, y=y_smooth, mode="lines",
        line=dict(color=color, width=width),
        name=label if show_legend else None,
        showlegend=show_legend
    ))

# layout
fig.update_layout(template="plotly_white",
                  title=dict(text=chart_title, x=0.5),
                  xaxis_title=x_axis_title,
                  yaxis_title=y_axis_title,
                  xaxis_type="log", width=1000, height=600,
                  legend=dict(x=1, y=1, xanchor="left", yanchor="top"))
fig.update_xaxes(showline=True, linewidth=axis_line_width, showticklabels=show_xaxis)
fig.update_yaxes(showline=True, linewidth=axis_line_width, showticklabels=show_yaxis)
if manual_xlim and x_min is not None and x_max is not None:
    fig.update_xaxes(range=[np.log10(x_min), np.log10(x_max)])
if manual_ylim and y_min is not None and y_max is not None:
    fig.update_yaxes(range=[y_min, y_max])

st.plotly_chart(fig, use_container_width=True)

# ============================
# Per-curve configuration (after figure)
# ============================
st.markdown("----")
st.subheader("⚙️ Configuration par courbe (modifie le tracé immédiatement)")
for i, comp in enumerate(selected):
    st.write(f"**{comp}**")
    col1, col2, col3, col4 = st.columns([2,1,1,2])
    with col1:
        _ = st.text_input(f"Libellé légende ({comp})",
                          value=st.session_state.get(f"legend_text_{comp}", comp),
                          key=f"legend_text_{comp}")
    with col2:
        _ = st.selectbox("Modèle", options=["4PL","3PL","Polynomial 2nd"],
                         index=["4PL","3PL","Polynomial 2nd"].index(st.session_state.get(f"model_{comp}", "4PL")),
                         key=f"model_{comp}")
    with col3:
        _ = st.selectbox("Forcer bornes", options=["Aucune","0%","100%","0+100%"],
                         index=["Aucune","0%","100%","0+100%"].index(st.session_state.get(f"force_{comp}", "Aucune")),
                         key=f"force_{comp}")
    with col4:
        _ = st.color_picker("Couleur", value=st.session_state.get(f"color_{comp}", palette[i % len(palette)]),
                            key=f"color_{comp}")

    c1,c2 = st.columns([1,1])
    with c1:
        _ = st.slider("Épaisseur ligne", 1, 6,
                      value=st.session_state.get(f"width_{comp}", global_line_width),
                      key=f"width_{comp}")
    with c2:
        _ = st.checkbox("Afficher points ± SD",
                        value=st.session_state.get(f"points_{comp}", True),
                        key=f"points_{comp}")

# ============================
# Export
# ============================
st.sidebar.markdown("---")
st.sidebar.header("💾 Export")
try:
    png_buf = BytesIO(fig.to_image(format="png", scale=3))
    st.sidebar.download_button("Exporter PNG", data=png_buf.getvalue(), file_name="HTRF_interactive.png", mime="image/png")
except Exception:
    st.sidebar.info("Export PNG nécessite kaleido (pip install kaleido).")

summary_rows = [{"Composé": c["comp"], "Modèle": c["model"], "EC50": c["ec50"], "Params": c["popt"]} for c in curve_data]
df_summary = pd.DataFrame(summary_rows)
csv_buf = BytesIO()
csv_buf.write(df_summary.to_csv(index=False).encode())
st.sidebar.download_button("Exporter CSV (résumé)", data=csv_buf.getvalue(), file_name="HTRF_interactive_summary.csv", mime="text/csv")

st.subheader("📊 Tableau récapitulatif")
st.dataframe(df_summary.style.format({"EC50": "{:.2f}"}))
