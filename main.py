import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import json
import matplotlib.colors as mcolors
import matplotlib.cm as cm
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="🌍 Carbon Analytics Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🌱"
)

# Style CSS personnalisé
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Asap:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Asap', sans-serif !important;
    }
    
    .main {
        font-family: 'Asap', sans-serif !important;
    }
    
    .stApp {
        font-family: 'Asap', sans-serif !important;
    }
    
    .stMarkdown {
        font-family: 'Asap', sans-serif !important;
    }
    
    .stText {
        font-family: 'Asap', sans-serif !important;
    }
    
    .stSelectbox {
        font-family: 'Asap', sans-serif !important;
    }
    
    .stMultiselect {
        font-family: 'Asap', sans-serif !important;
    }
    
    .big-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: #43A640;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Asap', sans-serif !important;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        font-family: 'Asap', sans-serif !important;
    }
    
    .metric-card {
        background: #FFF286;
        padding: 1.2rem;
        border-radius: 15px;
        color: black;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(67, 166, 64, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        font-family: 'Asap', sans-serif !important;
        min-height: 140px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        font-family: 'Asap', sans-serif !important;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-family: 'Asap', sans-serif !important;
    }
    
    .section-header {
        background: #F8F8F8;
        color: black;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(248, 248, 248, 0.3);
        font-family: 'Asap', sans-serif !important;
    }
    
    .insight-box {
        background: #FFF286;
        padding: 1.5rem;
        border-radius: 15px;
        color: black;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(255, 242, 134, 0.3);
        font-family: 'Asap', sans-serif !important;
    }
    
    .stSelectbox > div > div > select {
        border: 2px solid #43A640;
        border-radius: 10px;
        font-weight: 500;
        font-family: 'Asap', sans-serif !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #FFF286 !important;
    }
    
    .css-1lcbmhc {
        background-color: #FFF286 !important;
    }
    
    .css-17eq0hr {
        background-color: #FFF286 !important;
    }
    
    .css-1rs6os {
        background-color: #FFF286 !important;
    }
    
    .css-1cypcdb {
        background-color: #FFF286 !important;
    }
    
    .css-1d391kg .css-1lcbmhc {
        background-color: #FFF286 !important;
    }
    
    .css-1d391kg .css-17eq0hr {
        background-color: #FFF286 !important;
    }
    
    .css-1d391kg .css-1rs6os {
        background-color: #FFF286 !important;
    }
    
    .css-1d391kg .css-1cypcdb {
        background-color: #FFF286 !important;
    }
    
    /* Additional sidebar selectors for better coverage */
    [data-testid="stSidebar"] {
        background-color: #FFF286 !important;
    }
    
    [data-testid="stSidebar"] > div {
        background-color: #FFF286 !important;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        background-color: #FFF286 !important;
    }
    
    [data-testid="stSidebar"] .css-1lcbmhc {
        background-color: #FFF286 !important;
    }
    
    [data-testid="stSidebar"] .css-17eq0hr {
        background-color: #FFF286 !important;
    }
    
    [data-testid="stSidebar"] .css-1rs6os {
        background-color: #FFF286 !important;
    }
    
    [data-testid="stSidebar"] .css-1cypcdb {
        background-color: #FFF286 !important;
    }
</style>
""", unsafe_allow_html=True)

# Configuration matplotlib/seaborn pour un style moderne
plt.style.use('default')
sns.set_palette("husl")
sns.set_style("whitegrid", {
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.bottom": False,
    "ytick.left": False,
})

# Configuration des couleurs
COLORS = {
    'primary': '#43A640',
    'secondary': '#FFF286',
    'success': '#43A640',
    'warning': '#FFF286',
    'danger': '#FF6B6B',
    'info': '#4ECDC4',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

CATEGORY_COLORS = ['#43A640', '#FFF286', '#FF6B6B', '#4ECDC4', '#A8E6CF']

@st.cache_data
def load_data():
    df = pd.read_csv("empreinte_carbone_data.csv", parse_dates=["date_participation"])
    return df

def create_matplotlib_pie_chart(data, title=""):
    """Crée un graphique en anneau (donut chart) avec matplotlib"""
    fig, ax = plt.subplots(figsize=(6, 5))
    # Use global CATEGORY_COLORS for pie slices
    colors = CATEGORY_COLORS * (len(data) // len(CATEGORY_COLORS) + 1)
    
    wedges, texts, autotexts = ax.pie(
        data.values(), 
        labels=data.keys(), 
        autopct='%1.1f%%',
        colors=colors[:len(data)],
        startangle=90,
        explode=[0.05] * len(data),  # Sépare légèrement les sections
        shadow=False,
        textprops={'fontsize': 12, 'fontweight': 'bold'},
        wedgeprops={'width': 0.4},  # Donut chart effect
        pctdistance=0.8  # Place numbers on the ring
    )
    # Style moderne
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

def create_matplotlib_bar_chart(data, title="", horizontal=True, color_gradient=True):
    """Crée un graphique en barres avec matplotlib, utilisant une échelle continue basée sur la couleur principale de l'app."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a continuous colormap from white to the main primary color
    primary_color = COLORS['primary']
    cmap = mcolors.LinearSegmentedColormap.from_list('main_gradient', ["#F4F8F3", primary_color])
    norm = mcolors.Normalize(vmin=min(data.values), vmax=max(data.values))
    bar_colors = [cmap(norm(val)) for val in data.values]
    
    if horizontal:
        bars = ax.barh(range(len(data)), data.values, color=bar_colors)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data.index)
        ax.set_xlabel('Tonnes CO₂/an', fontweight='bold')
        for i, (bar, value) in enumerate(zip(bars, data.values)):
            ax.text(value + 0.1, i, f'{value:.2f}', va='center', ha='left', fontweight='bold')
    else:
        bars = ax.bar(range(len(data)), data.values, color=bar_colors)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data.index, rotation=45, ha='right')
        ax.set_ylabel('Tonnes CO₂/an', fontweight='bold')
        for bar, value in zip(bars, data.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig

def create_seaborn_violin_plot(df, x_col, y_col, hue_col, title=""):
    """Crée un graphique violon avec seaborn"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.violinplot(
        data=df, 
        x=x_col, 
        y=y_col, 
        hue=hue_col,
        palette="husl",
        inner="quart",
        ax=ax
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(ax.get_xlabel(), fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.legend(title=hue_col.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def create_seaborn_heatmap(corr_matrix, title=""):
    """Crée une heatmap de corrélation avec seaborn"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap='RdBu_r',
        center=0,
        square=True,
        fmt='.2f',
        cbar_kws={"shrink": .8},
        ax=ax
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

# Interface principale
def main():
    # En-tête
    st.markdown('<div class="big-title">🌍 Carbon Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">📊 Dashboard analytique des empreintes carbone</div>', unsafe_allow_html=True)
    
    # Chargement des données
    df = load_data()
    
    # Sidebar avec style
    st.sidebar.markdown("## 🎛️ Filtres de données")
    
    selected_city = st.sidebar.multiselect(
        "🏙️ Ville",
        sorted(df['city'].unique())
    )
    
    selected_logement_type = st.sidebar.multiselect(
        "🏠 Type de logement", 
        df['logement_type'].unique()
    )
    
    selected_nb_habitants = st.sidebar.multiselect(
        "👥 Nombre d'habitants", 
        sorted(df['logement_nb_habitants'].unique())
    )
    
    selected_type_chauffage = st.sidebar.multiselect(
        "🔥 Type de chauffage", 
        df['logement_type_chauffage'].unique()
    )
    
    # Filtrage des données
    mask = pd.Series([True] * len(df), index=df.index)
    
    if len(selected_logement_type) > 0:
        mask &= df['logement_type'].isin(selected_logement_type)
    
    if len(selected_nb_habitants) > 0:
        mask &= df['logement_nb_habitants'].isin(selected_nb_habitants)
    
    if len(selected_type_chauffage) > 0:
        mask &= df['logement_type_chauffage'].isin(selected_type_chauffage)
    
    if len(selected_city) > 0:
        mask &= df['city'].isin(selected_city)
    
    filtered_df = df[mask]
    
    # Harmonize city names for the choropleth map
    filtered_df['city'] = filtered_df['city'].replace({'Bruxelles (centre-ville)': 'Bruxelles'})
    
    # =====================
    # GLOBAL ANALYTICS (TOP)
    # =====================
    st.markdown('<div class="section-header">📊 Métriques principales</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("📝", "Réponses", f"{len(filtered_df):,}"),
        ("🌍", "Empreinte moyenne", f"{filtered_df['impact_total'].mean():.2f} t CO₂"),
        ("🔸", "Empreinte médiane", f"{filtered_df['impact_total'].median():.2f} t CO₂"),
        ("🏙️", "Communes représentées", f"{filtered_df['city'].nunique()}") ,
    ]
    for col, (icon, label, value) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f'''
            <div class="metric-card">
                <div style="font-size: 2rem;">{icon}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            ''', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Choropleth Map
    st.markdown('<div class="section-header">🗺️ Carte choroplèthe : Impact par commune</div>', unsafe_allow_html=True)
    # --- NEW: Sélection du type de métrique et de l'agrégation ---
    metric_options = {
        'Impact total': 'impact_total',
        'Logement': 'impact_logement',
        'Transport': 'impact_transport',
        'Alimentation': 'impact_alimentation',
        'Vêtements': 'impact_vetements',
        'Divertissement': 'impact_divertissement',
    }
    agg_options = {
        'Moyenne': 'mean',
        'Somme': 'sum',
        'Médiane': 'median',
    }
    col_metric, col_agg = st.columns(2)
    with col_metric:
        selected_metric_label = st.selectbox(
            "Sélectionnez la métrique à visualiser",
            list(metric_options.keys()),
            index=0
        )
        selected_metric = metric_options[selected_metric_label]
    with col_agg:
        selected_agg_label = st.selectbox(
            "Type d'agrégation",
            list(agg_options.keys()),
            index=0
        )
        selected_agg = agg_options[selected_agg_label]
    # --- END NEW ---
    
    with open("limites-administratives-des-communes-en-region-de-bruxelles-capitale.geojson", "r", encoding="utf-8") as f:
        brussels_geojson = json.load(f)
    
    # Aggregate selected metric by city
    if selected_agg == 'sum':
        city_metric = filtered_df.groupby('city')[selected_metric].sum().reset_index()
    elif selected_agg == 'mean':
        city_metric = filtered_df.groupby('city')[selected_metric].mean().reset_index()
    elif selected_agg == 'median':
        city_metric = filtered_df.groupby('city')[selected_metric].median().reset_index()
    else:
        city_metric = filtered_df.groupby('city')[selected_metric].sum().reset_index()
    
    # Plotly Choropleth
    fig_choropleth = px.choropleth_mapbox(
        city_metric,
        geojson=brussels_geojson,
        locations='city',
        featureidkey="properties.name_fr",
        color=selected_metric,
        color_continuous_scale="YlOrRd",
        mapbox_style="carto-positron",
        zoom=10.5,
        center={"lat": 50.8503, "lon": 4.3517},
        opacity=0.8,
        labels={selected_metric: f"{selected_metric_label} ({'t CO₂'})"}
    )
    # Remove borders for a modern look
    fig_choropleth.update_traces(marker_line_width=0)
    fig_choropleth.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        title_text=f"{selected_metric_label} par commune de Bruxelles ({selected_agg_label})",
        height=700
    )
    st.plotly_chart(fig_choropleth, use_container_width=True)
    
    # Pie Chart: Répartition par catégorie
    st.markdown('<div class="section-header">🥧 Répartition par catégorie</div>', unsafe_allow_html=True)
    category_means = {
        'Logement': filtered_df['impact_logement'].mean(),
        'Transport': filtered_df['impact_transport'].mean(),
        'Alimentation': filtered_df['impact_alimentation'].mean(),
        'Vêtements': filtered_df['impact_vetements'].mean(),
        'Divertissement': filtered_df['impact_divertissement'].mean()
    }
    fig_pie = create_matplotlib_pie_chart(
        category_means, 
        "Répartition moyenne de l'empreinte carbone"
    )
    st.pyplot(fig_pie)

    # Correlation Matrix
    st.markdown('<div class="section-header">🔗 Matrice de corrélation</div>', unsafe_allow_html=True)
    numeric_cols = [
        'logement_superficie_m2', 'logement_nb_habitants', 'logement_temperature_chauffage',
        'transport_km_voiture', 'transport_nb_vols_avion', 'alimentation_repas_viande_semaine',
        'impact_logement', 'impact_transport', 'impact_alimentation', 'impact_vetements', 
        'impact_divertissement', 'impact_total'
    ]
    corr_matrix = filtered_df[numeric_cols].corr()
    fig_heatmap = create_seaborn_heatmap(corr_matrix, "Corrélations entre variables")
    st.pyplot(fig_heatmap)

    # Timeline
    st.markdown('<div class="section-header">📅 Évolution temporelle</div>', unsafe_allow_html=True)
    filtered_df['mois_participation'] = filtered_df['date_participation'].dt.to_period('M')
    monthly_responses = filtered_df.groupby('mois_participation').agg({
        'impact_total': 'mean',
        'date_participation': 'count'
    }).reset_index()
    monthly_responses['mois_participation'] = monthly_responses['mois_participation'].astype(str)
    fig_timeline = make_subplots(specs=[[{"secondary_y": True}]])
    fig_timeline.add_trace(
        go.Scatter(
            x=monthly_responses['mois_participation'], 
            y=monthly_responses['date_participation'], 
            name="Nombre de réponses",
            line=dict(color=COLORS['primary'], width=3)
        ),
        secondary_y=False,
    )
    fig_timeline.add_trace(
        go.Scatter(
            x=monthly_responses['mois_participation'], 
            y=monthly_responses['impact_total'], 
            name="Empreinte moyenne",
            line=dict(color=COLORS['danger'], width=3)
        ),
        secondary_y=True,
    )
    fig_timeline.update_xaxes(title_text="Mois")
    fig_timeline.update_yaxes(title_text="Nombre de réponses", secondary_y=False)
    fig_timeline.update_yaxes(title_text="Empreinte moyenne (t CO₂)", secondary_y=True)
    fig_timeline.update_layout(
        title_text="Réponses et empreinte moyenne au fil du temps",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Key Insights
    st.markdown('<div class="section-header">💡 Insights clés</div>', unsafe_allow_html=True)
    max_logement = filtered_df.groupby('logement_type')['impact_total'].mean()
    min_logement = max_logement.min()
    max_logement_name = max_logement.idxmax()
    max_logement_value = max_logement.max()
    min_logement_name = max_logement.idxmin()
    dominant_category = max(category_means, key=category_means.get)
    dominant_percentage = category_means[dominant_category] / sum(category_means.values()) * 100
    st.markdown(f'''
    <div class="insight-box">
        <h3>🎯 Faits saillants</h3>
        <ul>
            <li><strong>{len(filtered_df):,}</strong> réponses analysées</li>
            <li>Empreinte moyenne: <strong>{filtered_df['impact_total'].mean():.2f} t CO₂/an</strong></li>
            <li>Logement le plus émetteur: <strong>{max_logement_name}</strong> ({max_logement_value:.2f} t CO₂/an)</li>
            <li>Logement le moins émetteur: <strong>{min_logement_name}</strong> ({min_logement:.2f} t CO₂/an)</li>
            <li>Catégorie dominante: <strong>{dominant_category}</strong> ({dominant_percentage:.1f}% de l'empreinte totale)</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)

    # =====================
    # DETAILED ANALYTICS (END)
    # =====================

    # --- HOUSING ---
    st.markdown('<div class="section-header">🏠 Détail Logement</div>', unsafe_allow_html=True)
    # Mini-metrics for Housing
    colh1, colh2 = st.columns(2)
    most_common_isolation = filtered_df['logement_isolation'].mode()[0]
    avg_home_size_per_person = filtered_df['logement_superficie_m2'].mean() / filtered_df['logement_nb_habitants'].mean()
    with colh1:
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 1.5rem;">🧱</div>
            <div class="metric-value">{most_common_isolation}</div>
            <div class="metric-label">Isolation la plus courante</div>
        </div>
        ''', unsafe_allow_html=True)
    with colh2:
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 1.5rem;">📏</div>
            <div class="metric-value">{avg_home_size_per_person:.1f} m²/pers</div>
            <div class="metric-label">Superficie moyenne par personne</div>
        </div>
        ''', unsafe_allow_html=True)
    st.markdown("#### 📊 Impact par type de logement")
    logement_avg = filtered_df.groupby('logement_type')['impact_total'].mean().sort_values(ascending=True)
    fig_bar = create_matplotlib_bar_chart(
        logement_avg,
        "Empreinte totale moyenne par type de logement",
        horizontal=True
    )
    st.pyplot(fig_bar)

    # --- TRANSPORT ---
    st.markdown('<div class="section-header">🚗 Détail Transport</div>', unsafe_allow_html=True)
    # Mini-metric for Transport
    colt1 = st.columns(1)[0]
    most_common_inhabitants = filtered_df['logement_nb_habitants'].mode()[0]
    with colt1:
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 1.5rem;">👥</div>
            <div class="metric-value">{most_common_inhabitants}</div>
            <div class="metric-label">Nombre d'habitants le plus courant</div>
        </div>
        ''', unsafe_allow_html=True)
    st.markdown("#### 🔀 Relation entre km parcourus en voiture et impact transport")
    fig_joint = sns.jointplot(
        data=filtered_df,
        x='transport_km_voiture',
        y='impact_transport',
        kind='hex',
        height=7,
        marginal_kws=dict(bins=20, fill=True)
    )
    fig_joint.fig.suptitle("Relation entre km parcourus en voiture et impact transport", fontsize=16, fontweight='bold', y=1.02)
    st.pyplot(fig_joint.fig)

    # --- MOBILITY ---
    st.markdown('<div class="section-header">🚲 Détail Mobilité</div>', unsafe_allow_html=True)
    st.markdown("#### 🚲 Impact moyen par mode de transport doux")
    mobility_modes = ['transport_km_velo', 'transport_km_velo_electrique', 'transport_km_trottinette']
    mobility_impact = filtered_df[mobility_modes].mean()
    fig_mobility = create_matplotlib_bar_chart(
        mobility_impact,
        "Distance moyenne annuelle par mode de transport doux (km)",
        horizontal=True
    )
    st.pyplot(fig_mobility)

    # --- FOOD ---
    st.markdown('<div class="section-header">🍽️ Détail Alimentation</div>', unsafe_allow_html=True)
    # Mini-metric for Food
    colf1 = st.columns(1)[0]
    avg_meat_meals = filtered_df['alimentation_repas_viande_semaine'].mean()
    with colf1:
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 1.5rem;">🍖</div>
            <div class="metric-value">{avg_meat_meals:.1f}</div>
            <div class="metric-label">Repas viande/semaine (moyenne)</div>
        </div>
        ''', unsafe_allow_html=True)
    st.markdown("#### 🍖 Impact alimentation selon repas viande/semaine")
    food_impact = filtered_df.groupby('alimentation_repas_viande_semaine')['impact_alimentation'].mean().sort_index()
    fig_food = create_matplotlib_bar_chart(
        food_impact,
        "Impact alimentation moyen selon nombre de repas viande/semaine",
        horizontal=False
    )
    st.pyplot(fig_food)

    # --- ENTERTAINMENT ---
    st.markdown('<div class="section-header">🎮 Détail Divertissement</div>', unsafe_allow_html=True)
    # Mini-metrics for Entertainment
    cole1, cole2 = st.columns(2)
    most_common_quality = filtered_df['divertissement_qualite_video'].mode()[0]
    avg_devices = filtered_df['divertissement_appareils_numeriques'].mean()
    with cole1:
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 1.5rem;">📺</div>
            <div class="metric-value">{most_common_quality}</div>
            <div class="metric-label">Qualité vidéo la plus courante</div>
        </div>
        ''', unsafe_allow_html=True)
    with cole2:
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 1.5rem;">💻</div>
            <div class="metric-value">{avg_devices:.1f}</div>
            <div class="metric-label">Appareils numériques (moyenne)</div>
        </div>
        ''', unsafe_allow_html=True)
    st.markdown("#### 📺 Impact divertissement par qualité vidéo")
    entertainment_impact = filtered_df.groupby('divertissement_qualite_video')['impact_divertissement'].mean().sort_index()
    fig_entertainment = create_matplotlib_bar_chart(
        entertainment_impact,
        "Impact divertissement moyen par qualité vidéo",
        horizontal=True
    )
    st.pyplot(fig_entertainment)

    # Bar chart: Impact by daily video hours
    st.markdown("#### ⏰ Impact divertissement selon les heures de vidéo par jour")
    entertainment_by_hours = filtered_df.groupby('divertissement_heures_video_jour')['impact_divertissement'].mean()
    fig_hours = create_matplotlib_bar_chart(
        entertainment_by_hours,
        "Impact moyen selon les heures de vidéo par jour",
        horizontal=True
    )
    st.pyplot(fig_hours)

    # Data Preview
    st.markdown('<div class="section-header">📋 Aperçu des données</div>', unsafe_allow_html=True)
    st.dataframe(filtered_df.head(50), use_container_width=True)
if __name__ == "__main__":
    main()
