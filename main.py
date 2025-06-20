import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Set page config
st.set_page_config(page_title="Carbon Footprint Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

# Load data from CSV
@st.cache_data
def load_data():
    df = pd.read_csv("empreinte_carbone_data.csv", parse_dates=["date_participation"])
    return df

df = load_data()

# Dashboard title
st.title("🌍 Carbon Footprint Analytics Dashboard")
st.markdown("### Analyse des empreintes carbone des utilisateurs (données réelles)")

# Sidebar filters (using available columns)
st.sidebar.header("Filtres")
selected_logement_type = st.sidebar.multiselect(
    "Type de logement", df['logement_type'].unique(), default=df['logement_type'].unique())
selected_nb_habitants = st.sidebar.multiselect(
    "Nombre d'habitants", sorted(df['logement_nb_habitants'].unique()), default=sorted(df['logement_nb_habitants'].unique()))
selected_type_chauffage = st.sidebar.multiselect(
    "Type de chauffage", df['logement_type_chauffage'].unique(), default=df['logement_type_chauffage'].unique())

# Filter data
filtered_df = df[
    (df['logement_type'].isin(selected_logement_type)) &
    (df['logement_nb_habitants'].isin(selected_nb_habitants)) &
    (df['logement_type_chauffage'].isin(selected_type_chauffage))
]

# Key metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Nombre de réponses", f"{len(filtered_df):,}")
with col2:
    st.metric("Empreinte totale moyenne", f"{filtered_df['impact_total'].mean():.2f} t CO₂")
with col3:
    st.metric("Types de logement", filtered_df['logement_type'].nunique())
with col4:
    st.metric("Types de chauffage", filtered_df['logement_type_chauffage'].nunique())
with col5:
    st.metric("Superficie moyenne", f"{filtered_df['logement_superficie_m2'].mean():.1f} m²")

# Main charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Empreinte par catégorie")
    category_means = {
        'Logement': filtered_df['impact_logement'].mean(),
        'Transport': filtered_df['impact_transport'].mean(),
        'Alimentation': filtered_df['impact_alimentation'].mean(),
        'Vêtements': filtered_df['impact_vetements'].mean(),
        'Divertissement': filtered_df['impact_divertissement'].mean()
    }
    fig_pie = px.pie(
        values=list(category_means.values()),
        names=list(category_means.keys()),
        title="Répartition moyenne de l'empreinte carbone"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("Empreinte totale par type de logement")
    logement_avg = filtered_df.groupby('logement_type')['impact_total'].mean().sort_values(ascending=True)
    fig_bar = px.bar(
        x=logement_avg.values,
        y=logement_avg.index,
        orientation='h',
        title="Empreinte totale moyenne par type de logement",
        labels={'x': 'Tonnes CO₂/an', 'y': 'Type de logement'}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Detailed analysis
st.subheader("Analyse détaillée par catégorie")
col1, col2 = st.columns(2)

with col1:
    st.write("**Empreinte par type de chauffage**")
    chauffage_categories = filtered_df.melt(
        id_vars=['logement_type_chauffage'],
        value_vars=['impact_logement', 'impact_transport', 'impact_alimentation', 'impact_vetements', 'impact_divertissement'],
        var_name='catégorie',
        value_name='empreinte'
    )
    chauffage_categories['catégorie'] = chauffage_categories['catégorie'].str.replace('impact_', '').str.title()
    fig_chauffage = px.box(
        chauffage_categories,
        x='logement_type_chauffage',
        y='empreinte',
        color='catégorie',
        title="Distribution de l'empreinte carbone par type de chauffage"
    )
    st.plotly_chart(fig_chauffage, use_container_width=True)

with col2:
    st.write("**Empreinte par nombre d'habitants**")
    habitants_categories = filtered_df.melt(
        id_vars=['logement_nb_habitants'],
        value_vars=['impact_logement', 'impact_transport', 'impact_alimentation', 'impact_vetements', 'impact_divertissement'],
        var_name='catégorie',
        value_name='empreinte'
    )
    habitants_categories['catégorie'] = habitants_categories['catégorie'].str.replace('impact_', '').str.title()
    fig_habitants = px.violin(
        habitants_categories,
        x='logement_nb_habitants',
        y='empreinte',
        color='catégorie',
        title="Distribution de l'empreinte carbone par nombre d'habitants"
    )
    st.plotly_chart(fig_habitants, use_container_width=True)

# Time series analysis
st.subheader("Chronologie des réponses")
filtered_df['mois_participation'] = filtered_df['date_participation'].dt.to_period('M')
monthly_responses = filtered_df.groupby('mois_participation').agg({
    'impact_total': 'mean',
    'date_participation': 'count'
}).reset_index()
monthly_responses['mois_participation'] = monthly_responses['mois_participation'].astype(str)
fig_timeline = make_subplots(specs=[[{"secondary_y": True}]])
fig_timeline.add_trace(
    go.Scatter(x=monthly_responses['mois_participation'], y=monthly_responses['date_participation'], name="Nombre de réponses"),
    secondary_y=False,
)
fig_timeline.add_trace(
    go.Scatter(x=monthly_responses['mois_participation'], y=monthly_responses['impact_total'], name="Empreinte moyenne"),
    secondary_y=True,
)
fig_timeline.update_xaxes(title_text="Mois")
fig_timeline.update_yaxes(title_text="Nombre de réponses", secondary_y=False)
fig_timeline.update_yaxes(title_text="Empreinte moyenne (t CO₂)", secondary_y=True)
fig_timeline.update_layout(title_text="Réponses et empreinte moyenne au fil du temps")
st.plotly_chart(fig_timeline, use_container_width=True)

# Data table
st.subheader("Aperçu des données brutes")
st.dataframe(filtered_df.head(100), use_container_width=True)

# Summary insights
st.subheader("Faits saillants")
st.write(f"""
**Aperçu du jeu de données :**
- Nombre total de réponses analysées : {len(filtered_df):,}
- Empreinte carbone moyenne : {filtered_df['impact_total'].mean():.2f} t CO₂/an
- Type de logement le plus émetteur : {filtered_df.groupby('logement_type')['impact_total'].mean().idxmax()} ({filtered_df.groupby('logement_type')['impact_total'].mean().max():.2f} t CO₂/an)
- Type de logement le moins émetteur : {filtered_df.groupby('logement_type')['impact_total'].mean().idxmin()} ({filtered_df.groupby('logement_type')['impact_total'].mean().min():.2f} t CO₂/an)

**Catégories dominantes :**
- Le logement représente {category_means['Logement'] / sum(category_means.values()) * 100:.1f}% de l'empreinte totale moyenne
""")