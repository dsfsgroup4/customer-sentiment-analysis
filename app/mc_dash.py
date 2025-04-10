import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from streamlit_extras.switch_page_button import switch_page


# Titre du dashboard
st.title("Visualisation des avis McDonald's")


@st.cache_data
def load_data(file_path):
    """Charge le fichier CSV et retourne un DataFrame."""
    return pd.read_csv(file_path, encoding='latin-1')

# Charger les données
df = load_data("df2_model_SIA.csv")

# Ajouter une option pour afficher les données brutes
if st.checkbox('Afficher les données brutes'):
    st.subheader('Données brutes')
    st.write(df)


def analyze_sentiment(df):
    # Appliquer l'analyse de sentiment
    df['sentiment'] = df['review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return df

def display_top_reviews(df):
    # Trier les avis
    top_positive = df["review"][df.sentiment_SIA == "positive"].head() #(5, 'sentiment')  # Top 5 positifs
    top_negative = df["review"][df.sentiment_SIA == "negative"].head()  # Top 5 négatifs

    st.subheader("🔥 Top 5 positive reviews")
    for i, row in top_positive.iterrows():
        st.write(f"⭐ {row['review']} (Sentiment: {row['sentiment_SIA']:.2f})")

    st.subheader("❄️ Top 5 des avis négatifs")
    for i, row in top_negative.iterrows():
        st.write(f"⚠️ {row['review']} (Sentiment: {row['sentiment_SIA']:.2f})")

def display_rating_distribution(df):
    # Vérifier si la colonne 'rating' existe
    if 'ratings' in df.columns:
        # Compter le nombre d'avis pour chaque rating
        rating_counts = df['sentiment_SIA'].value_counts().sort_index()

        # Afficher l’histogramme
        st.subheader("📊 Distribution des Ratings")
        st.bar_chart(rating_counts)
    else:
        st.error("❌ La colonne 'rating' est absente du dataset.")

def display_restaurant_details(df):
    # Vérifier si la colonne 'store_address' existe
    if 'store_address' in df.columns:
        # Liste des adresses uniques
        address_list = df['store_address'].unique()

        # Ajout du filtre dans la barre latérale
        selected_address = st.sidebar.selectbox("📍 Sélectionnez un restaurant", ["Tous"] + list(address_list))

        # Bouton pour afficher les détails du restaurant
        if selected_address != "Tous":
            if st.sidebar.button("🔍 Voir les détails"):
                # Sauvegarder l'adresse sélectionnée en session
                st.session_state["selected_address"] = selected_address
                # Filtrer les données du restaurant sélectionné
                selected_restaurant_data = df[df['store_address'] == selected_address]
                st.session_state["restaurant_data"] = selected_restaurant_data
                st.session_state["selected_restaurant"] = selected_restaurant_data['store_address'].iloc[0]
                st.rerun()  # Recharger la page pour passer aux détails
    else:
        st.error("❌ La colonne 'store_address' est absente du dataset.")

def display_selected_restaurant():
    # Vérifier que l'adresse sélectionnée existe dans la session
    if "selected_address" in st.session_state and st.session_state["selected_address"]:
        selected_address = st.session_state["selected_address"]
        selected_restaurant = st.session_state["selected_restaurant"]
        restaurant_data = st.session_state["restaurant_data"]

        # Afficher les informations du restaurant
        st.title(f"🏢 Détails du restaurant : {selected_restaurant}")

        # Ajouter une case à cocher pour afficher ou cacher les détails
        show_details = st.checkbox("📍 Afficher les détails du restaurant")

        if show_details:
            st.subheader(f"📍 Adresse : {selected_address}")
            st.dataframe(restaurant_data)


        if "show_details" not in st.session_state:
            st.session_state["show_details"] = False
        if "show_reviews" not in st.session_state:
            st.session_state["show_reviews"] = False

 # Bouton pour afficher/cacher les avis clients
        if st.button("💬 Voir/Masquer les avis clients"):
            st.session_state["show_reviews"] = not st.session_state["show_reviews"]

            # Afficher/cacher les avis clients si nécessaire
        if st.session_state["show_reviews"]:
            st.subheader("💬 Avis des clients")
            for review in restaurant_data["review"].dropna().head(5):
                st.write(f"🗨️ {review}")
        
# Afficher l'histogramme des ratings pour ce restaurant
        st.subheader("📊 Distribution des Ratings")
        rating_counts = restaurant_data['ratings'].value_counts().sort_index()
        st.bar_chart(rating_counts)



        # Afficher les avis du restaurant
        #st.subheader("💬 Avis des clients")
        #for review in restaurant_data["review"].dropna().head(5):
        #    st.write(f"🗨️ {review}")
        
        # Bouton pour revenir à la page principale
        if st.button("⬅️ Retour à la liste"):
            st.session_state["selected_address"] = None  # Clear the session state
            st.session_state["selected_restaurant"] = None
            st.rerun()  # Re-run to return to the main page

    else:
        st.error("❌ Aucun restaurant sélectionné. Pour afficher les avis clients à propos de votre restaurant veuillez en choisir un à gauche de l'écran.")


# Vérifier et analyser les sentiments si possible
if 'review' in df.columns:
    df = df.dropna(subset=['review'])
    df = analyze_sentiment(df)
    display_top_reviews(df)

# Afficher la distribution des ratings
display_rating_distribution(df)

# Afficher les détails du restaurant si nécessaire
display_restaurant_details(df)


# Afficher les détails du restaurant sélectionné
display_selected_restaurant()