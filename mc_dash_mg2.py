import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
import pydeck as pdk
import branca.colormap as cm
from branca.element import Template, MacroElement

# Titre du dashboard
st.title("Client's reviews dashboard")

@st.cache_data
def load_data(file_path):
    """Charge le fichier CSV et retourne un DataFrame."""
    df = pd.read_csv(file_path, encoding='latin-1')
    return df

# Charger les données
df = load_data("df4_model_roberta_pour_dash.csv")

# Ajout des filtres dans la barre latérale
st.sidebar.header("Select a location")
# Liste des States uniques
state_list = df['State'].unique()
selected_state = st.sidebar.selectbox("📍 Select your State", ["All"] + list(state_list))

# Filtrer les villes en fonction de l'état sélectionné
if selected_state == "All":
    city_list = df['City'].unique()
else:
    city_list = df[df['State'] == selected_state]['City'].unique()
selected_city = st.sidebar.selectbox("📍 Select your City", ["All"] + list(city_list))

# Filtrer les adresses en fonction de la ville et de l'état sélectionnés
if selected_state == "All" and selected_city == "All":
    address_list = df['store_address'].unique()
elif selected_state != "All" and selected_city == "All":
    address_list = df[df['State'] == selected_state]['store_address'].unique()
elif selected_state == "All" and selected_city != "All":
    address_list = df[df['City'] == selected_city]['store_address'].unique()
else:
    address_list = df[(df['State'] == selected_state) & (df['City'] == selected_city)]['store_address'].unique()
selected_address = st.sidebar.selectbox("📍 Select your restaurant", ["All"] + list(address_list))


# Sidebar pour sélectionner une période
st.sidebar.header("Select a period")
# Sélection des dates de début et de fin
df['review_date'] = pd.to_datetime(df['review_date'])
start_date = st.sidebar.date_input("Starting date", value=df['review_date'].min(), min_value=df['review_date'].min(), max_value=df['review_date'].max())
end_date = st.sidebar.date_input("End date", value=df['review_date'].max(), min_value=df['review_date'].min(), max_value=df['review_date'].max())

# Filtrer le DataFrame en fonction de la période, de l'état, de la ville et de l'adresse sélectionnés
filtered_df = df[(df['review_date'] >= pd.to_datetime(start_date)) & (df['review_date'] <= pd.to_datetime(end_date))]
if selected_state != "All":
    filtered_df = filtered_df[filtered_df['State'] == selected_state]
if selected_city != "All":
    filtered_df = filtered_df[filtered_df['City'] == selected_city]
if selected_address != "All":
    filtered_df = filtered_df[filtered_df['store_address'] == selected_address]


if selected_address == "All":
    st.markdown(f"""
    <div style='line-height: 1.2;'>
        <span style='font-size: 2em; font-weight: bold;'>National Sentiment score overview</span><br>
        <span style='font-size: 1.5em;'>From {start_date} to {end_date}</span><br>
        <br>
    </div>
    """, unsafe_allow_html=True)
else:
    selected_restaurant = filtered_df.iloc[0]['store_address']
    st.markdown(f"""
    <div style='line-height: 1.2;'>
        <span style='font-size: 2em; font-weight: bold;'>Sentiment score overview at :<br>
        📍 {selected_restaurant}</span><br>
        <span style='font-size: 1.5em;'>From {start_date} to {end_date}</span><br>
        <br>
    </div>
    """, unsafe_allow_html=True)
    

# Ajouter une option pour afficher les données brutes
if st.checkbox('Display raw data'):
    st.subheader('Raw data')
    st.write(filtered_df)

# Ajouter une option pour afficher le describe des données brutes
if st.checkbox('Display data info'):
    st.subheader('Data info')
    st.write(filtered_df.describe(include="all"))

#### CREATE 4 COLUMNS
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_nb_review = len(filtered_df['review'])
    st.markdown(f"""
        <div style='line-height: 1.2;'>
            <span style='font-size: 1.5em; font-weight: bold;'>{total_nb_review}</span><br>
            <span style='font-size: 1em;'>Number of reviews</span>
        </div>
    """, unsafe_allow_html=True)

with col2:
    total_negative = len(filtered_df['review'][filtered_df["RoBerta_label"] == "negative"])
    st.markdown(f"""
        <div style='line-height: 1.2;'>
            <span style='font-size: 1.5em; font-weight: bold;'>🤬{total_negative}</span><br>
            <span style='color: red; font-weight: bold;'>Negative</span>
        </div>
    """, unsafe_allow_html=True)

with col3:
    total_neutral = len(filtered_df['review'][filtered_df["RoBerta_label"] == "neutral"])
    st.markdown(f"""
        <div style='line-height: 1.2;'>
            <span style='font-size: 1.5em; font-weight: bold;'>😐️{total_neutral}</span><br>
            <span style='color: yellow; font-weight: bold;'>Neutral</span>
        </div>
    """, unsafe_allow_html=True)

with col4:
    total_positive = len(filtered_df['review'][filtered_df["RoBerta_label"] == "positive"])
    st.markdown(f"""
        <div style='line-height: 1.2;'>
            <span style='font-size: 1.5em; font-weight: bold;'>😍{total_positive}</span><br>
            <span style='color: green; font-weight: bold;'>Positive</span>
        </div>
    """, unsafe_allow_html=True)

#affichage de la carte 
if selected_address == "All":
    if 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns and 'RoBerta_score' in filtered_df.columns and 'store_address' in filtered_df.columns:
        # Calculer le RoBerta_score moyen pour chaque localisation
        filtered_df['RoBerta_score'] = pd.to_numeric(filtered_df['RoBerta_score'], errors='coerce')
        filtered_df_grouped = filtered_df.groupby(['latitude', 'longitude', 'store_address']).agg({'RoBerta_score': 'mean'}).reset_index()

        # Créer une carte centrée sur les États-Unis
        m = folium.Map(location=[37.0902, -95.7129], zoom_start=4, tiles="cartodb Dark Matter")

        # Fonction pour obtenir la couleur en fonction du score
        def get_color(score):
            # Couleurs distinctes : rouge, orange, jaune, vert clair, vert
            if score <= 0.745978: #min
                # Rouge
                return "#FF0000"
            elif score <= 0.791968: #Q1
                # Orange
                return "#FFA500"
            elif score <= 0.806491: #Q2
                # Jaune
                return "#FFFF00"
            elif score <= 0.822416: #Q3
                # Vert clair
                return "#90EE90"
            else:
                # Vert
                return "#008000"

        # Ajouter des marqueurs pour chaque paire de coordonnées
        for _, row in filtered_df_grouped.iterrows():
            if pd.notnull(row['latitude']) and pd.notnull(row['longitude']):
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=10,
                    color=get_color(row['RoBerta_score']),
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"{row['store_address']}<br>Score: {row['RoBerta_score']:.2f}"
                ).add_to(m)

        # Créer une légende
        colormap = cm.LinearColormap(
            colors=['#FF0000', '#FFA500', '#FFFF00', '#90EE90', '#008000'],
            vmin=0.745978,
            vmax=0.822416,
            caption='Sentiment Score'
        )

        # Ajouter la légende à la carte
        m.add_child(colormap)

        # Afficher la carte dans Streamlit
        folium_static(m)
        max_score = filtered_df["RoBerta_score"].max()
        min_score = filtered_df["RoBerta_score"].min()
        st.markdown(f"""
            <span style='font-weight: bold;'>Average sentiment score</span><br>
            <span style='font-style: italic;'>From 0 (very negative) in red to 1 (very positive) in green</span><br>
            <span>Min = {min_score:.2f} &nbsp;&nbsp;&nbsp; Max = {max_score:.2f}</span>
        """, unsafe_allow_html=True)

    else:
        st.error("Les colonnes 'latitude', 'longitude', 'RoBerta_score' ou 'store_address' sont absentes du dataset.")
else:
    st.markdown(f"""
                <br>
    """, unsafe_allow_html=True)



#### CREATE 2 COLUMNS for positive and negative reviews
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
        <span style='font-size: 1.5em; font-weight: bold;'>🔥Top Positive subjects</span><br>
    """, unsafe_allow_html=True)

    top_positive = filtered_df[filtered_df["RoBerta_label"] == "positive"].sort_values(by='RoBerta_score', ascending=False) 

    def display_top_positive_reviews(filtered_df):  
        for index, (review_index, review) in enumerate(top_positive[["review", "review_time"]].head(5).iterrows()):
            review_text = review["review"]
            review_time = review["review_time"]
            st.write(f"💬{review_text} - <i>{review_time}</i>", unsafe_allow_html=True)  # Afficher l'avis complet avec la date en italique
    
    if st.checkbox('Show top positive reviews'):
        display_top_positive_reviews(filtered_df)

with col2:
    st.markdown(f"""
        <span style='font-size: 1.5em; font-weight: bold;'>❄️ Top Negative subjects</span><br>
    """, unsafe_allow_html=True) 
    top_negative = filtered_df[filtered_df["RoBerta_label"] == "negative"].sort_values(by='RoBerta_score', ascending=True)

    def display_top_negative_reviews(filtered_df):  
        for index, (review_index, review) in enumerate(top_negative[["review", "review_time"]].head(5).iterrows()):
            review_text = review["review"]
            review_time = review["review_time"]
            st.write(f"💬{review_text} - <i>{review_time}</i>", unsafe_allow_html=True)  # Afficher l'avis complet avec la date en italique

    if st.checkbox('Show top negative reviews'): 
        display_top_negative_reviews(filtered_df)

