# Pour faire tourner le fichier app.py, il faut utiliser le resultats du modèle
# RoBERTa qui se trouve dans le dossier data

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path 
import matplotlib.pyplot as plt
import seaborn as sns

# ============================== CONFIG ==============================
DATA_PATH = Path("data\data_avec_labels.csv")

# ============================== PAGE SETUP ==============================
st.set_page_config(layout="wide", page_title="Restaurant Review Dashboard", page_icon="📊")

st.markdown("""
    <div style='padding-left: 10px; padding-right: 10px;'>
        <h3>Mcdonald's Dashboard</h3>
    </div>
""", unsafe_allow_html=True)

# ============================== LABELISATION =============================
# Rappel de la liste des labels utilisées par le model
labels = [
    'hygiene', 'food quality', 'food', 'staff', 'something is missing',
    'location', 'speed of service', 'drive-thru', 'temperature of the food',
    'atmosphere', 'customer service', "temperature" , "price", "speed", "quality", "courtesy",
]

# ============================== LOAD DATA ==============================
@st.cache_data
def load_data(path: Path):
    """Load review data from a CSV file with basic preprocessing."""
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()  # Clean column names
        df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
        df["review_date"] = pd.to_datetime(df["review_date"])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# ============================== FILTER DATA ==============================
def apply_filters(df):
    """Apply hierarchical location and date filters via the sidebar."""
    st.sidebar.header("📍 Location Filters")

    df["review_date"] = pd.to_datetime(df["review_date"])

    # === Select State ===
    state_list = sorted(df["State"].dropna().unique())
    selected_state = st.sidebar.selectbox("Select a State", ["All"] + state_list)

    # === Select City (based on State) ===
    if selected_state == "All":
        city_list = sorted(df["City"].dropna().unique())
    else:
        city_list = sorted(df[df["State"] == selected_state]["City"].dropna().unique())
    selected_city = st.sidebar.selectbox("Select a City", ["All"] + city_list)

    # === Select Restaurant Address (based on State & City) ===
    if selected_state == "All" and selected_city == "All":
        address_list = sorted(df["store_address"].dropna().unique())
    elif selected_city == "All":
        address_list = sorted(df[df["State"] == selected_state]["store_address"].dropna().unique())
    elif selected_state == "All":
        address_list = sorted(df[df["City"] == selected_city]["store_address"].dropna().unique())
    else:
        address_list = sorted(df[(df["State"] == selected_state) & (df["City"] == selected_city)]["store_address"].dropna().unique())
    selected_address = st.sidebar.selectbox("Select a Restaurant", ["All"] + address_list)

    # === Date Filter ===
    st.sidebar.header("📆 Select a Period")
    min_date = df["review_date"].min()
    max_date = df["review_date"].max()

    # Apply filters
    start_date = st.sidebar.date_input(
        "Starting date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
)
    end_date = st.sidebar.date_input(
    "End date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)
    # Protect against bad input (e.g. end before start)
    if start_date > end_date:
        st.sidebar.error("❌ End date must be after start date.")
        return pd.DataFrame()

    filtered_df = df[(df["review_date"] >= pd.to_datetime(start_date)) & (df["review_date"] <= pd.to_datetime(end_date))]

    if selected_state != "All":
        filtered_df = filtered_df[filtered_df["State"] == selected_state]
    if selected_city != "All":
        filtered_df = filtered_df[filtered_df["City"] == selected_city]
    if selected_address != "All":
        filtered_df = filtered_df[filtered_df["store_address"] == selected_address]

    # Store filters in session
    st.session_state["selected_filters"] = {
        "state": selected_state,
        "city": selected_city,
        "address": selected_address,
        "start_date": start_date,
        "end_date": end_date
    }

    return filtered_df


# ============================== UI HELPERS ==============================

def render_metric(label, value, bg_color, text_color):
    """
    Render a custom metric box.
    Accepts both numbers and strings (like percentages).
    """
    st.markdown(f"""
        #### {label}
        <div style='background-color:{bg_color}; padding: 0.7rem; border-radius: 6px; text-align:center; font-size:30px; color:{text_color};'>
            {value if isinstance(value, str) else f"{value:,}"}
        </div>
    """, unsafe_allow_html=True)


# Seuil pour filtrer les labels
seuil = 0.2

def render_comments(comments, color_primary, color_secondary):
    """Render a list of comments in stylized boxes."""
    for index, comment in comments.items():
        # Récupérer les labels avec un score supérieur au seuil pour ce commentaire
        labels_above_threshold = df[labels].loc[index] > seuil
        selected_labels = df[labels].columns[labels_above_threshold].tolist()
        labels_str = " ".join([f"#{label}" for label in selected_labels])
        formatted_comment = comment.replace('\n', ' ')
        st.markdown(f"<p style='color:{color_primary};'>💬 {formatted_comment}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{color_secondary}; font-weight: bold;'>{labels_str}</p>", unsafe_allow_html=True)

# ============================== MAIN APP ==============================
with st.spinner("Loading data..."):
    df = load_data(DATA_PATH)

if df.empty:
    st.warning("No data available. Please check the source file.")
    st.stop()

filtered_df = apply_filters(df)

dashboard_tab, reviews_tab = st.tabs(["📊 Overview", "📈 Review Trends"])

# ============================== NPS: Calcul des scores promoteurs / détracteurs ==============================
# -----  -----
def compute_nps_value(rating_int):
    if rating_int in [4, 5]:
        return 1
    elif rating_int == 3:
        return 0
    elif rating_int in [1, 2]:
        return -1
    else:
        return 0

filtered_df["nps_value"] = filtered_df["rating_int"].apply(compute_nps_value)

# Calcul des % par catégorie
promoters_pct = (filtered_df["nps_value"] == 1).mean() * 100
detractors_pct = (filtered_df["nps_value"] == -1).mean() * 100
passives_pct = (filtered_df["nps_value"] == 0).mean() * 100

# NPS Global
nps_score = promoters_pct - detractors_pct



# ============================== METRICS ==============================
with dashboard_tab:
    total_reviews = len(filtered_df)
    pos_reviews = len(filtered_df[filtered_df["pred_sentiment"] == "positive"])
    neg_reviews = len(filtered_df[filtered_df["pred_sentiment"] == "negative"])
    neu_reviews = len(filtered_df[filtered_df["pred_sentiment"] == "neutral"])


    # ----- CALCUL ET AFFICHAGE DU NPS (APPROCHE PAR POURCENTAGE) -----

# 1. Définir la fonction pour convertir une note en valeur NPS
    def compute_nps_value(rating_int):
        """
        Convertit une note (1 à 5) en une valeur NPS.
        - 4 ou 5 -> +1 (Promoteur)
        - 3     ->  0 (Passif)
        - 1 ou 2 -> -1 (Détracteur)
        """
        if rating_int in [4, 5]:
            return 1
        elif rating_int == 3:
            return 0
        elif rating_int in [1, 2]:
            return -1
        else:
            return 0

    # 2. Appliquer la fonction pour créer une colonne 'nps_value'
    filtered_df["nps_value"] = filtered_df["rating_int"].apply(compute_nps_value)

    # 3. Calculer les pourcentages
    promoters_pct = (filtered_df["nps_value"] == 1).mean() * 100
    detractors_pct = (filtered_df["nps_value"] == -1).mean() * 100
    passives_pct = (filtered_df["nps_value"] == 0).mean() * 100

    # 4. Affichage des métriques
    total_reviews = len(filtered_df)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        render_metric("📊 Total Reviews", total_reviews, "#2E3B4E", "white")
    with metric_col2:
        render_metric("🙂 Promoteurs (%)", f"{promoters_pct:.1f}%", "#1e3d2f", "#b7f7d0")
    with metric_col3:
        render_metric("😠 Détracteurs (%)", f"{detractors_pct:.1f}%", "#3d1e1e", "#ffb6b6")
    with metric_col4:
        render_metric("😐 Passifs (%)", f"{passives_pct:.1f}%", "#444444", "#eeeeee")


# ============================== AFFICHAGE DU SCORE NPS GLOBAL ==============================
    nps_color = "#1e3d2f" if nps_score > 50 else "#f9b233" if nps_score > 0 else "#7a0000"
    nps_text_color = "#ffffff"

    st.markdown("### 🧮 NPS Global Score")
    nps_col = st.columns(1)[0]
    with nps_col:
        render_metric("NPS Score", f"{nps_score:.1f}", nps_color, nps_text_color)


    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    st.divider()

# ============================== LOCATION MAP AND WEEKLY TRENDS ==============================
    st.subheader("🗺️ Mcdonald's US Map")
    map_col = st.columns(1)[0]

    with map_col:
        if {"latitude", "longitude", "City", "store_address", "rating_int"}.issubset(filtered_df.columns):
            # Préparer les données avec coordonnées uniques + avis + NPS
            location_df = filtered_df.dropna(subset=["latitude", "longitude", "store_address"])

            # Calcul des nps_value
            location_df["nps_value"] = location_df["rating_int"].apply(compute_nps_value)

            # Grouper par restaurant (coordonnées + adresse)
            map_data = location_df.groupby(["latitude", "longitude", "store_address"]).agg(
                review_count=("review", "count"),
                nps_score=("nps_value", lambda x: (x == 1).mean()*100 - (x == -1).mean()*100)
            ).reset_index()

            if not map_data.empty:
                fig_map = px.scatter_geo(
                    map_data,
                    lat="latitude",
                    lon="longitude",
                    size="review_count",
                    color="nps_score",
                    color_continuous_scale="RdYlGn",
                    hover_name="store_address",
                    size_max=30,
                    scope="usa",
                    template="plotly_dark",
                    title="Localisation des restaurants (taille = avis, couleur = NPS)",
                    height=450
                )
                fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=10))
                st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("Aucune donnée de localisation disponible après filtrage.")
        else:
            st.info("Données de localisation incomplètes.")


    st.divider()

    
# ============================== NPS PAR VILLE ============================== 
    st.markdown("### 🏙️ NPS par Ville")

    if "City" in filtered_df.columns and not filtered_df["City"].isna().all():
        nps_by_city = filtered_df.groupby("City").apply(
            lambda d: (d["nps_value"] == 1).mean()*100 - (d["nps_value"] == -1).mean()*100
        ).reset_index(name="NPS")

        fig_nps = px.bar(
            nps_by_city,
            x="City",
            y="NPS",
            color="NPS",
            color_continuous_scale="RdYlGn",
            title="NPS par Ville",
            template="plotly_dark",
            height=400
        )
        fig_nps.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_nps, use_container_width=True)
    else:
        st.info("Aucune donnée disponible pour afficher le NPS par ville.")

    st.divider()


# ============================== SENTIMENT TREND ROW ==============================
    st.subheader("Sentiment Trends")
    trends_row = st.columns(1)[0]

    with trends_row:
        df_trends = filtered_df[filtered_df["review_date"].notna()].copy()

        # Load current filters from session state
        filters = st.session_state.get("selected_filters", {})
        start = pd.to_datetime(filters.get("start_date", df_trends["review_date"].min()))
        end = pd.to_datetime(filters.get("end_date", df_trends["review_date"].max()))

        # Ensure valid date types
        if pd.isna(start): start = df_trends["review_date"].min()
        if pd.isna(end): end = df_trends["review_date"].max()

        # Filter by date
        df_trends = df_trends[(df_trends["review_date"] >= start) & (df_trends["review_date"] <= end)]

        # Group by day and sentiment
        df_trends["day"] = df_trends["review_date"].dt.date
        sentiment_counts = df_trends.groupby(["day", "pred_sentiment"]).size().reset_index(name="count")

        # Optional debug
        # st.write("Filtered trend size:", sentiment_counts.shape)

        # Dynamic subtitle based on filters
        selected_scope = "National" if filters.get("address") == "All" else f"📍 {filters.get('address')}"
        trend_subtitle = f"**Sentiment Trend for:** {selected_scope} — _{start.strftime('%b %d, %Y')} to {end.strftime('%b %d, %Y')}_"
        st.markdown(trend_subtitle)

        # Plot
        if sentiment_counts.empty:
            st.info("No sentiment data available for the selected period.")
        else:
            fig_sentiment = px.line(
                sentiment_counts,
                x="day",
                y="count",
                color="pred_sentiment",
                markers=True,
                title=None,
                template="plotly_dark",
                color_discrete_map={"positive": "green", "neutral": "orange", "negative": "red"},
                hover_data={"count": True}
            )
            st.plotly_chart(fig_sentiment, use_container_width=True, config={'displayModeBar': False})

        st.divider()
    
# ============================== TOP TOPICS ==============================
    

    topics_col1, topics_col2 = st.columns(2)

    with topics_col1:
        st.markdown("### 😍 Top good topics")
        positive_df = filtered_df[filtered_df['pred_sentiment'] == 'positive']
        top_topics = (positive_df[labels] > seuil).sum().sort_values(ascending=False).head(10)        

        plt.figure(figsize=(10, 15))
        sns.barplot(x=top_topics.values, y=top_topics.index, palette=["green"])

        # Ajuster la taille du texte
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)

        # Ajouter des annotations
        for i, v in enumerate(top_topics.values):
            plt.text(v + 0.1, i, str(v), color='black', va='center', fontsize=25)

        # Retirer l'encadrement du graphique
        plt.box(False)

        # Retirer l'axe horizontal
        plt.gca().xaxis.set_visible(False)
        # Supprimer l'étiquette de l'axe vertical
        plt.gca().set_ylabel('')
        
        # Afficher le graphique dans Streamlit
        st.pyplot(plt)

    with topics_col2:
        st.markdown("### 🤬 Worst bad topics")
        negative_df = filtered_df[filtered_df['pred_sentiment'] == 'negative']
        top_topics = (negative_df[labels] > seuil).sum().sort_values(ascending=False).head(10)        

        plt.figure(figsize=(10, 15))
        sns.barplot(x=top_topics.values, y=top_topics.index, palette=["red"])

        # Ajuster la taille du texte
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)

        # Ajouter des annotations
        for i, v in enumerate(top_topics.values):
            plt.text(v + 0.1, i, str(v), color='black', va='center', fontsize=25)

        # Retirer l'encadrement du graphique
        plt.box(False)

        # Retirer l'axe horizontal
        plt.gca().xaxis.set_visible(False)
        # Supprimer l'étiquette de l'axe vertical
        plt.gca().set_ylabel('')
        
        # Afficher le graphique dans Streamlit
        st.pyplot(plt)
    
# ============================== TOP COMMENTS ==============================
    # Initialiser l'état de la session pour le suivi des index de départ et des pages
    if 'positive_start_index' not in st.session_state:
        st.session_state.positive_start_index = 0
    if 'negative_start_index' not in st.session_state:
        st.session_state.negative_start_index = 0
    if 'positive_page' not in st.session_state:
        st.session_state.positive_page = 1
    if 'negative_page' not in st.session_state:
        st.session_state.negative_page = 1

    comment_col1, comment_col2 = st.columns(2)

    # Palette de couleurs par sentiment
    sentiment_styles = {
        "positive": {"bg": "#b7f7d0", "text": "#b7f7d0", "label": "👍 Positive Comments"},
        "neutral": {"bg": "#4b4b1e", "text": "#f9eec0", "label": "😐 Neutral Comments"},
        "negative": {"bg": "#ffb6b6", "text": "#ffb6b6", "label": "👎 Negative Comments"},
    }

    with comment_col1:
        style = sentiment_styles["positive"]
        st.markdown(f"#### {style['label']}")

        # Ajouter un menu déroulant pour sélectionner un sujet
        selected_top_topic = st.selectbox("Select a topic to filter the good comments", options=["All"] + labels, key="positive_topic")

        # Filtrer le DataFrame en fonction du sujet sélectionné
        if selected_top_topic != "All":
            topic_filtered_df = filtered_df[filtered_df[selected_top_topic] > seuil]
        else:
            topic_filtered_df = filtered_df

        top_pos_df = topic_filtered_df[topic_filtered_df["pred_sentiment"] == "positive"].sort_values(by='RoBERTa_score', ascending=False)
        top_pos = top_pos_df["review"].iloc[st.session_state.positive_start_index:st.session_state.positive_start_index+5]
        render_comments(top_pos, style["bg"], style["text"])

        # Afficher le numéro de la page
        st.write(f"Page {st.session_state.positive_page}")

        # Ajouter des boutons pour charger les commentaires précédents et suivants
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.session_state.positive_page > 1:
                if st.button("Previous", key="positive_prev_button"):
                    st.session_state.positive_start_index -= 5
                    st.session_state.positive_page -= 1
                    st.rerun()
        with col2:
            if st.session_state.positive_start_index + 5 < len(top_pos_df):
                if st.button("Next", key="positive_next_button"):
                    st.session_state.positive_start_index += 5
                    st.session_state.positive_page += 1
                    st.rerun()

    with comment_col2:
        style = sentiment_styles["negative"]
        st.markdown(f"#### {style['label']}")

        # Ajouter un menu déroulant pour sélectionner un sujet
        selected_bad_topic = st.selectbox("Select a topic to filter the bad comments", options=["All"] + labels, key="negative_topic")

        # Filtrer le DataFrame en fonction du sujet sélectionné
        if selected_bad_topic != "All":
            topic_filtered_df = filtered_df[filtered_df[selected_bad_topic] > seuil]
        else:
            topic_filtered_df = filtered_df

        top_neg_df = topic_filtered_df[topic_filtered_df["pred_sentiment"] == "negative"].sort_values(by='RoBERTa_score', ascending=False)
        top_neg = top_neg_df["review"].iloc[st.session_state.negative_start_index:st.session_state.negative_start_index+5]
        render_comments(top_neg, style["bg"], style["text"])

        # Afficher le numéro de la page
        st.write(f"Page {st.session_state.negative_page}")

        # Ajouter des boutons pour charger les commentaires précédents et suivants
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.session_state.negative_page > 1:
                if st.button("Previous", key="negative_prev_button"):
                    st.session_state.negative_start_index -= 5
                    st.session_state.negative_page -= 1
                    st.rerun()
        with col2:
            if st.session_state.negative_start_index + 5 < len(top_neg_df):
                if st.button("Next", key="negative_next_button"):
                    st.session_state.negative_start_index += 5
                    st.session_state.negative_page += 1
                    st.rerun()
# ============================== EMPTY STATE ==============================
if filtered_df.empty:
    st.warning("No reviews match the selected filters. Try adjusting them.")
