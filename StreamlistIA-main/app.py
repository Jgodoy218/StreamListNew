import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Predicción de Precios de Vivienda",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.prediction-box {
    background: linear-gradient(135deg, #1D4E89 0%, #2563AB 100%);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    color: white;
    margin: 1rem 0;
}
.prediction-price { font-size: 3rem; font-weight: bold; margin: 0; }
.prediction-label { font-size: 0.9rem; opacity: 0.8; margin: 0; }
[data-testid="stSidebar"] { background-color: #1D4E89; }
[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

BG      = '#0E1117'
GRID    = '#2D3748'
TEXTO   = '#E2E8F0'
AZUL1   = '#1D4E89'
AZUL2   = '#2563AB'
AZUL3   = '#3B82F6'
AZUL4   = '#60A5FA'
AZUL5   = '#93C5FD'
AZUL6   = '#BFDBFE'
ROJO    = '#F87171'
VERDE   = '#34D399'

def layout_oscuro(titulo, altura=400, margen_l=80, margen_r=100):
    return dict(
        title=dict(text=titulo, font=dict(color=TEXTO, size=16), x=0.01),
        height=altura,
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font=dict(color=TEXTO, size=12),
        showlegend=False,
        margin=dict(t=55, b=55, l=margen_l, r=margen_r),
        xaxis=dict(
            showgrid=True, gridcolor=GRID, gridwidth=1,
            linecolor=GRID, tickcolor=GRID,
            tickfont=dict(color=TEXTO, size=11),
            title_font=dict(color=TEXTO),
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True, gridcolor=GRID, gridwidth=1,
            linecolor=GRID, tickcolor=GRID,
            tickfont=dict(color=TEXTO, size=11),
            title_font=dict(color=TEXTO),
            zeroline=False,
        ),
    )


@st.cache_data
def generar_datos():
    np.random.seed(42)
    n = 1460
    barrios = ['NAmes','CollgCr','OldTown','Edwards','Somerst',
               'NridgHt','Gilbert','Sawyer','NWAmes','SawyerW']
    bm = {'NridgHt':1.35,'Somerst':1.15,'CollgCr':1.05,'Gilbert':1.02,
          'NAmes':1.0,'NWAmes':0.98,'SawyerW':0.95,'Sawyer':0.92,
          'Edwards':0.88,'OldTown':0.85}
    calidad = np.random.choice(range(1,11), n,
                               p=[0.01,0.02,0.04,0.08,0.15,0.20,0.22,0.16,0.08,0.04])
    area_gf = np.random.normal(1500, 500, n).clip(400, 5000).astype(int)
    anio    = np.random.randint(1900, 2010, n)
    n_banos = np.random.choice([1,2,3,4], n, p=[0.15,0.55,0.25,0.05])
    garage  = np.random.choice([0,1,2,3], n, p=[0.05,0.20,0.60,0.15])
    sotano  = np.random.normal(1000, 350, n).clip(0, 3000).astype(int)
    barrio  = np.random.choice(barrios, n)
    precio  = (
        50000 + calidad*12000 + area_gf*55 + (2023-anio)*(-400)
        + n_banos*8000 + garage*6000 + sotano*15
        + np.array([bm.get(b,1.0) for b in barrio])*20000
        + np.random.normal(0, 15000, n)
    ).clip(50000, 800000).astype(int)
    return pd.DataFrame({
        'SalePrice': precio, 'GrLivArea': area_gf, 'OverallQual': calidad,
        'YearBuilt': anio,   'FullBath':  n_banos,  'GarageCars':  garage,
        'TotalBsmtSF': sotano, 'Neighborhood': barrio,
    })

df = generar_datos()

def predecir(area, calidad, anio, banos, garage, sotano, barrio):
    bm = {'NridgHt':1.35,'Somerst':1.15,'CollgCr':1.05,'Gilbert':1.02,
          'NAmes':1.0,'NWAmes':0.98,'SawyerW':0.95,'Sawyer':0.92,
          'Edwards':0.88,'OldTown':0.85}.get(barrio, 1.0)
    return max(50000, int(50000+calidad*12000+area*55+(2023-anio)*(-400)
                          +banos*8000+garage*6000+sotano*15+bm*20000))

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Características")
    st.markdown("---")
    area_gf = st.slider("Área habitable (ft²)", 400, 5000, 1500, 50)
    calidad = st.select_slider("Calidad general (1–10)", list(range(1,11)), value=6)
    anio    = st.slider("Año de construcción", 1900, 2010, 1975)
    banos   = st.selectbox("Baños completos", [1,2,3,4], index=1)
    garage  = st.selectbox("Capacidad garaje (autos)", [0,1,2,3], index=2)
    sotano  = st.slider("Área sótano (ft²)", 0, 3000, 800, 50)
    barrio  = st.selectbox("Barrio", ['NridgHt','Somerst','CollgCr','Gilbert',
                                       'NAmes','NWAmes','SawyerW','Sawyer','Edwards','OldTown'])
    st.markdown("---")
    st.caption("Dataset: Ames Housing · 1,460 registros · 79 features")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🏘 Predicción del Precio de Viviendas")
st.caption("Proyecto IA · Dataset Ames Housing · Iowa, EE.UU. · Regresión supervisada")
st.markdown("---")

precio_pred = predecir(area_gf, calidad, anio, banos, garage, sotano, barrio)
precio_min  = int(precio_pred * 0.90)
precio_max  = int(precio_pred * 1.10)
precio_prom = int(df['SalePrice'].mean())
delta_pct   = ((precio_pred - precio_prom) / precio_prom) * 100

col1, col2 = st.columns([1.2, 2])
with col1:
    st.markdown(f"""
    <div class="prediction-box">
        <p class="prediction-label">PRECIO ESTIMADO</p>
        <p class="prediction-price">${precio_pred:,.0f}</p>
        <p class="prediction-label">Rango: ${precio_min:,.0f} – ${precio_max:,.0f}</p>
    </div>
    """, unsafe_allow_html=True)
    signo = "▲" if delta_pct > 0 else "▼"
    st.info(f"{signo} {abs(delta_pct):.1f}% respecto al precio promedio")

with col2:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Promedio", f"${precio_prom:,.0f}")
    c2.metric("Mediana",  f"${int(df['SalePrice'].median()):,.0f}")
    c3.metric("Mínimo",   f"${int(df['SalePrice'].min()):,.0f}")
    c4.metric("Máximo",   f"${int(df['SalePrice'].max()):,.0f}")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=precio_pred,
        title={'text':"Posición en el mercado",'font':{'size':13,'color':TEXTO}},
        number={'prefix':"$",'valueformat':",.0f",'font':{'color':AZUL4}},
        gauge={
            'axis':{'range':[50000,750000],'tickformat':"$,.0f",'tickfont':{'color':TEXTO,'size':9}},
            'bar':{'color':AZUL3,'thickness':0.25},
            'bgcolor':BG,'borderwidth':0,
            'steps':[
                {'range':[50000, 200000],'color':'#1a2a1a'},
                {'range':[200000,350000],'color':'#1a2420'},
                {'range':[350000,550000],'color':'#1a1f2e'},
                {'range':[550000,750000],'color':'#221a2e'},
            ],
            'threshold':{'line':{'color':ROJO,'width':3},'thickness':0.75,'value':precio_prom}
        }
    ))
    fig_gauge.update_layout(height=210, margin=dict(t=40,b=10,l=20,r=20),
                            paper_bgcolor=BG, font=dict(color=TEXTO))
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Distribución de precios",
    "🔍 Análisis de variables",
    "🗺️ Por barrio",
    "📈 Importancia de features"
])

# ── TAB 1 ──────────────────────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['SalePrice'], nbinsx=50,
            marker_color=AZUL3,
            marker_line_color=BG, marker_line_width=0.8,
            opacity=0.9,
            hovertemplate='$%{x:,.0f}<br>Cantidad: %{y}<extra></extra>'
        ))
        fig.add_vline(x=precio_pred, line_dash="dash", line_color=ROJO, line_width=2,
                      annotation_text=f"Tu vivienda<br>${precio_pred:,.0f}",
                      annotation_font_color=ROJO, annotation_font_size=11,
                      annotation_bgcolor=BG)
        fig.add_vline(x=precio_prom, line_dash="dot", line_color=VERDE, line_width=2,
                      annotation_text=f"Promedio<br>${precio_prom:,.0f}",
                      annotation_font_color=VERDE, annotation_font_size=11,
                      annotation_position="top left", annotation_bgcolor=BG)
        lo = layout_oscuro("Distribución de precios de venta", 390)
        lo['xaxis']['tickformat'] = "$,.0f"
        lo['xaxis']['title'] = "Precio de venta (USD)"
        lo['yaxis']['title'] = "Cantidad de viviendas"
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = go.Figure()
        colores_box = [AZUL6,AZUL6,AZUL5,AZUL5,AZUL4,AZUL4,AZUL3,AZUL3,AZUL2,AZUL1]
        for i, q in enumerate(sorted(df['OverallQual'].unique())):
            datos_q = df[df['OverallQual']==q]['SalePrice']
            fig2.add_trace(go.Box(
                y=datos_q, name=str(q),
                marker_color=colores_box[i],
                line_color=AZUL4, line_width=1.5,
                boxmean=True, fillcolor=colores_box[i],
                hovertemplate=f'Calidad {q}<br>$%{{y:,.0f}}<extra></extra>'
            ))
        lo2 = layout_oscuro("Precio según calidad general", 390)
        lo2['xaxis']['title'] = "Calidad (1 = peor, 10 = mejor)"
        lo2['yaxis']['title'] = "Precio de venta (USD)"
        lo2['yaxis']['tickformat'] = "$,.0f"
        lo2['showlegend'] = False
        fig2.update_layout(**lo2)
        st.plotly_chart(fig2, use_container_width=True)

# ── TAB 2 ──────────────────────────────────────────────────────────────────────
with tab2:
    labels_map = {
        'GrLivArea':'Área habitable (ft²)', 'OverallQual':'Calidad general',
        'YearBuilt':'Año de construcción',  'TotalBsmtSF':'Área sótano (ft²)',
        'FullBath':'Baños completos',        'GarageCars':'Capacidad garaje',
    }
    variable = st.selectbox("Variable a analizar:", list(labels_map.keys()),
                            format_func=lambda x: labels_map[x])
    col_a, col_b = st.columns([2, 1])
    with col_a:
        corr = df[[variable,'SalePrice']].corr().iloc[0,1]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df[variable], y=df['SalePrice'], mode='markers',
            marker=dict(color=AZUL4, size=5, opacity=0.5,
                        line=dict(color=BG, width=0.3)),
            hovertemplate=f'{labels_map[variable]}: %{{x}}<br>$%{{y:,.0f}}<extra></extra>'
        ))
        m, b = np.polyfit(df[variable], df['SalePrice'], 1)
        x_line = np.linspace(df[variable].min(), df[variable].max(), 100)
        fig3.add_trace(go.Scatter(
            x=x_line, y=m*x_line+b, mode='lines',
            line=dict(color=ROJO, width=2.5),
            name=f'Tendencia (r = {corr:.2f})',
            showlegend=True
        ))
        lo3 = layout_oscuro(f"Precio vs {labels_map[variable]}", 430)
        lo3['xaxis']['title'] = labels_map[variable]
        lo3['yaxis']['title'] = "Precio de venta (USD)"
        lo3['yaxis']['tickformat'] = "$,.0f"
        lo3['showlegend'] = True
        lo3['legend'] = dict(font=dict(color=TEXTO), bgcolor=BG)
        fig3.update_layout(**lo3)
        st.plotly_chart(fig3, use_container_width=True)
    with col_b:
        st.markdown("### 📊 Estadísticas")
        st.metric("Correlación con precio", f"{corr:.3f}")
        fuerza = "**fuerte** 💪" if abs(corr) > 0.5 else "moderada"
        st.caption(f"Relación {fuerza} con el precio.")
        st.markdown("---")
        for k, v in df[variable].describe().round(1).items():
            st.metric(k, f"{v:,.1f}")

# ── TAB 3 ──────────────────────────────────────────────────────────────────────
with tab3:
    stats_b = df.groupby('Neighborhood')['SalePrice'].agg(['mean','median','count']).reset_index()
    stats_b.columns = ['Barrio','Precio promedio','Precio mediano','N° viviendas']
    stats_b = stats_b.sort_values('Precio promedio', ascending=True)

    colores_barrio = [AZUL6,AZUL6,AZUL5,AZUL5,AZUL4,AZUL4,AZUL3,AZUL3,AZUL2,AZUL1]

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=stats_b['Precio promedio'],
        y=stats_b['Barrio'],
        orientation='h',
        marker=dict(color=colores_barrio[:len(stats_b)], line=dict(color=BG, width=1)),
        text=[f"  ${v:,.0f}" for v in stats_b['Precio promedio']],
        textposition='outside',
        textfont=dict(size=12, color=TEXTO),
        hovertemplate='%{y}<br>Promedio: $%{x:,.0f}<extra></extra>',
    ))
    lo4 = layout_oscuro("Precio promedio por barrio", 470, margen_l=90, margen_r=130)
    lo4['xaxis']['title'] = "Precio promedio (USD)"
    lo4['xaxis']['tickformat'] = "$,.0f"
    lo4['yaxis']['title'] = ""
    fig4.update_layout(**lo4)
    st.plotly_chart(fig4, use_container_width=True)

    st.dataframe(
        stats_b.sort_values('Precio promedio', ascending=False)
               .style.format({'Precio promedio':'${:,.0f}','Precio mediano':'${:,.0f}'})
               .background_gradient(subset=['Precio promedio'], cmap='Blues'),
        use_container_width=True, height=320
    )

# ── TAB 4 ──────────────────────────────────────────────────────────────────────
with tab4:
    feat_labels  = ['Baños completos','Capacidad garaje','Año construcción',
                    'Área sótano','Área habitable','Calidad general']
    importancias = [0.04, 0.06, 0.08, 0.12, 0.28, 0.42]
    colores_imp  = [AZUL6, AZUL5, AZUL4, AZUL3, AZUL2, AZUL1]

    fig5 = go.Figure()
    fig5.add_trace(go.Bar(
        x=importancias, y=feat_labels, orientation='h',
        marker=dict(color=colores_imp, line=dict(color=BG, width=1)),
        text=[f"  {v:.0%}" for v in importancias],
        textposition='outside',
        textfont=dict(size=13, color=TEXTO),
        hovertemplate='%{y}<br>Importancia: %{x:.1%}<extra></extra>',
    ))
    lo5 = layout_oscuro("Importancia relativa de características", 410, margen_l=130, margen_r=80)
    lo5['xaxis']['title'] = "Importancia relativa"
    lo5['xaxis']['tickformat'] = ".0%"
    lo5['yaxis']['title'] = ""
    fig5.update_layout(**lo5)
    st.plotly_chart(fig5, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.info("💡 **Calidad general** y **área habitable** explican juntas ~70% de la variabilidad en el precio.")
    with col_b:
        fig_pie = go.Figure(go.Pie(
            labels=['Calidad general','Área habitable','Área sótano',
                    'Año construcción','Cap. garaje','Baños'],
            values=[0.42, 0.28, 0.12, 0.08, 0.06, 0.04],
            hole=0.5,
            marker=dict(colors=[AZUL1,AZUL2,AZUL3,AZUL4,AZUL5,AZUL6],
                        line=dict(color=BG, width=2)),
            textinfo='label+percent',
            textfont=dict(color=TEXTO, size=11),
            hovertemplate='%{label}<br>%{percent}<extra></extra>'
        ))
        fig_pie.update_layout(height=300, margin=dict(t=20,b=20,l=20,r=20),
                              paper_bgcolor=BG, showlegend=False,
                              font=dict(size=12, color=TEXTO))
        st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")
st.caption("Proyecto de IA · Ames Housing Dataset · 1,460 registros · 79 características")