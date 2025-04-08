# Pour faire tourner le fichier app.py, il faut utiliser le fichier data_avec_labels.csv dans le dossier data

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================== CONFIG ==============================
DATA_PATH = Path("data/data_avec_labels.csv")

# ============================== PAGE SETUP ==============================
st.set_page_config(layout="wide", page_title="Restaurant Review Dashboard", page_icon="📊")

st.markdown("""
    <div style='padding-left: 10px; padding-right: 10px;'>
        <h1>Mcdonald's Dashboard</h1>
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
        <div style='
            background-color:{bg_color}; 
            border-radius: 50%; 
            display: flex;
            align-items: center;
            justify-content: center;
            text-align:center; 
            font-size:25px; 
            font-weight: bold;
            width: 150px;
            height: 150px;
            margin: auto;
            color:{text_color};'>
            {value if isinstance(value, str) else f"{value:,}"}
        </div>
    """, unsafe_allow_html=True)


# Seuil pour filtrer les labels
seuil = 0.2

def render_comments(title, comments, color_primary, color_secondary):
    """Render a list of comments in stylized boxes."""
    st.markdown(f"#### {title}")
    for index, comment in comments.items():
        # Récupérer les labels avec un score supérieur au seuil pour ce commentaire
        labels_above_threshold = df[labels].loc[index] > seuil
        selected_labels = df[labels].columns[labels_above_threshold].tolist()
        labels_str = " ".join([f"#{label}" for label in selected_labels])
        st.markdown(f"<p style='color:{color_primary};'>{comment}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{color_secondary};'>{labels_str}</p>", unsafe_allow_html=True)


# ============================== MAIN APP ==============================
with st.spinner("Loading data..."):
    df = load_data(DATA_PATH)

if df.empty:
    st.warning("No data available. Please check the source file.")
    st.stop()

filtered_df = apply_filters(df)

dashboard_tab, reviews_tab = st.tabs(["📊 Overview", "📈 Review Trends"])


# ============================== NPS MAPPING HELPER ==============================

def compute_nps_value(sentiment):
    """
    Convert sentiment to NPS value:
    - 'positive' => +1 (Promoter)
    - 'neutral'  =>  0 (Passive)
    - 'negative' => -1 (Detractor)
    """
    return {'positive': 1, 'neutral': 0, 'negative': -1}.get(sentiment, 0)


sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
filtered_df["nps_value"] = filtered_df["pred_sentiment"].map(sentiment_mapping).fillna(0)


# Calcul des % par catégorie
promoters_pct = (filtered_df["nps_value"] == 1).mean() * 100
detractors_pct = (filtered_df["nps_value"] == -1).mean() * 100
passives_pct = (filtered_df["nps_value"] == 0).mean() * 100

# NPS Global
nps_score = promoters_pct - detractors_pct

#liste the NPS max et minim
max_value_pd = filtered_df["nps_value"].max()* 100
min_value_pd = filtered_df["nps_value"].min()* 100


# ============================== METRICS ====================================================
with dashboard_tab:
    total_reviews = len(filtered_df)

    # ============================== AFFICHAGE DU SCORE NPS GLOBAL ==============================

    #====== Appliquer la fonction pour créer une colonne 'nps_value'==========
    filtered_df["nps_value"] = filtered_df["pred_sentiment"].apply(compute_nps_value)

    #========= Calculer les pourcentages ===============
    promoters_pct = (filtered_df["nps_value"] == 1).mean() * 100
    detractors_pct = (filtered_df["nps_value"] == -1).mean() * 100
    passives_pct = (filtered_df["nps_value"] == 0).mean() * 100

    #========= Affichage des métriques ==================
    total_reviews = len(filtered_df)
    nps_color = "#1aa442" if nps_score > 50 else "#b36500" if nps_score > 0 else "#aa0000"
    nps_text_color = "#ffffff"

    #========= Tooltip of NPS ==========================
    with st.expander("ℹ️ What is NPS?", expanded=False):
        st.markdown("""
            **Net Promoter Score (NPS)** measures customer loyalty by subtracting the percentage of detractors from promoters.
            
            - **Promoters** (positive): Loyal enthusiasts.
            - **Passives** (neutral): Satisfied but unenthusiastic.
            - **Detractors** (negative): Unhappy customers.

            **NPS = %Promoters - %Detractors**
        """)

    # ============== NPS Title =========================
    st.markdown("### 🧮 NPS Score ")
    st.markdown("<div  style='height: 0px;  display: flex; align-items: center;'>", unsafe_allow_html=True)

    total_col,nps_col, prom_col,detract_col, passif_col = st.columns(5)

    with total_col:
        render_metric("📊 Total Reviews", total_reviews, "#2E3B4E", "white")
    with nps_col:
        render_metric("NPS Score", f"{nps_score:.1f}", nps_color, nps_text_color)

    with prom_col:
        render_metric("🙂 Promoters", f"{promoters_pct:.1f}%", "#137830", "#b7f7d0")
    
    with detract_col:
        render_metric("😠 Detractors", f"{detractors_pct:.1f}%", "#aa0000", "#ffb6b6")
    with passif_col:
        render_metric("😐 Passives", f"{passives_pct:.1f}%", "#b36500", "#eeeeee")

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()


# ============================== LOCATION MAP AND WEEKLY TRENDS ==============================

    # ======= US MAP Tooltip =======
    with st.expander("ℹ️ About this Map"):
        st.markdown("""
        This map shows McDonald's locations across the US. 
        - **Bubble size** = Number of reviews
        - **Color** = NPS score (green = high, red = low)
        - Hover over bubbles to explore store details.
    """)

    # ============= MCdonalds US map ==============
    st.subheader("🗺️ Mcdonald's US Map")
    map_col = st.columns(1)[0]

    with map_col:
        if {"latitude", "longitude", "City", "store_address", "pred_sentiment"}.issubset(filtered_df.columns):
            # Préparer les données avec coordonnées uniques + avis + NPS
            location_df = filtered_df.dropna(subset=["latitude", "longitude", "store_address"])

            # Calcul des nps_value
            location_df["nps_value"] = location_df["pred_sentiment"].apply(compute_nps_value)


            # Grouper par restaurant (coordonnées + adresse)
            map_data = location_df.groupby(["latitude", "longitude", "store_address"]).agg(
                review_count=("clean_reviews", "count"),
                nps_score=("nps_value", lambda x: (x == 1).mean()*100 - (x == -1).mean()*100)
            ).reset_index()

            # Color scale is constant regardless of data
            fixed_nps_range = [min_value_pd, max_value_pd]
                                        
            # Define custom color gradient (red to white to green)
            custom_nps_scale = [
                [0.0, "red"],     # minimum NPS
                [0.4, "orange"],   # neutral
                [1.0, "green"]    # maximum NPS
]

            if not map_data.empty:
                fig_map = px.scatter_geo(
                    map_data,
                    lat="latitude",
                    lon="longitude",
                    size="review_count",
                    color="nps_score",
                    color_continuous_scale=custom_nps_scale,
                    range_color=fixed_nps_range,
                    hover_name="store_address",
                    hover_data={
                    "nps_score": ':.1f',
                    "review_count": True
                        },
                    size_max=30,
                    scope="usa",
                    template="plotly_dark",
                    title="Location of restaurants (size = reviews, color = NPS)",
                    height=450
                )
                fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=10))
                st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("No location data available after filtering.")
        else:
            st.info("Incomplete location data.")


    st.divider()

# ========================== NPS by restaurant Bar Chart ============================ 
    # ======= NPS by Restaurant Bar chart ==========
    with st.expander("ℹ️ About NPS Scores of restaurants Bar Chart"):
        st.markdown("""
        This chart shows Net Promoter Score (NPS) for each store based on filtered date and location.
        
        - **Higher bars** mean better customer sentiment.
        - Hover over a bar to get restaurant's info like review count, city, and state.
    """)

    # NPS Scores of restaurants' title
    st.markdown("### 🏙️ NPS Scores of restaurants")

    # Retrieve current filters from session stat
    filters = st.session_state.get("selected_filters", {})
    start = filters.get("start_date")
    end = filters.get("end_date")

    # Build dynamic title based on the "address" filter:
    selected_scope = "All Restaurants" if filters.get("address") == "All" else f"📍 {filters.get('address')}"
    nps_title = f"**NPS per Restaurant for:** {selected_scope} — _{start.strftime('%b %d, %Y')} to {end.strftime('%b %d, %Y')}_"

    # Display the dynamic title in the app
    st.markdown(nps_title)

    if "store_address" in filtered_df.columns and not filtered_df["store_address"].isna().all():
        # Group by restaurant and compute NPS score as the difference between the percentage of promoters and detractors
        nps_by_restaurant = (
            filtered_df.groupby("store_address")
            .agg(
                NPS=("nps_value", lambda x: (x == 1).mean() * 100 - (x == -1).mean() * 100),
                review_count=("nps_value", "count"),
                City=("City", "first"),
                State=("State", "first")
            )
            .reset_index()
            .sort_values("NPS", ascending=True)  # Sort from highest to lowest NPS
        )

        fig_nps = px.bar(
            nps_by_restaurant,
            x="NPS",
            y="store_address",
            orientation="h",
            color="NPS",
            color_continuous_scale=custom_nps_scale,
            hover_data={
                "NPS": ":.2f",
                "review_count": True,
                "City": True,
                "State": True
                        },
            template="plotly_dark",
            height=700
        )
        # Rotate x-axis labels to improve readability
        fig_nps.update_layout(xaxis_title="NPS Score",yaxis_title="Restaurants") 
        st.plotly_chart(fig_nps, use_container_width=True )
    else:
        st.info("Aucune donnée disponible pour afficher le NPS par ville.")

    st.divider()

# ============================== TOP COMMENTS ==============================
    st.markdown("### 🗨️ Top topics")
    with st.expander("ℹ️ What's This?"):
        st.markdown("""
        These are the most frequent topics mentioned in reviews.
        Based on review content and labels predicted.
    """)


    topics_col1, topics_col2 = st.columns(2)
    
    # ====== Bar graph de Top topic positive ============
    with topics_col1:
        positive_df = filtered_df[filtered_df['pred_sentiment'] == 'positive']
        top_topics = (positive_df[labels] > seuil).sum().sort_values(ascending=False).head(10)        

        fig_topics = go.Figure()

        fig_topics.add_trace(go.Bar(
            x=top_topics.values,
            y=top_topics.index,
            orientation='h',
            marker=dict(color='green'),
            text=top_topics.values,
            textposition='auto',
            hovertemplate='%{y}: %{x} mentions<extra></extra>',
        ))

        fig_topics.update_layout(
            title="Most Frequent Positive Topics",
            xaxis_title="Mention Count",
            yaxis_title="Topic",
            height=500,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20),
        )

        fig_topics.update_yaxes(autorange="reversed")  # Most frequent on top
        st.plotly_chart(fig_topics, use_container_width=True)


    
    comment_col1, comment_col2 = st.columns(2)

    # Palette de couleurs par sentiment
    sentiment_styles = {
        "positive": {"bg": "#b7f7d0", "text": "#1e3d2f", "label": "👍 Top Positive Comments"},
        "neutral": {"bg": "#4b4b1e", "text": "#f9eec0", "label": "😐 Neutral Comments"},
        "negative": {"bg": "#ffb6b6", "text": "#3d1e1e", "label": "👎 Top Negative Comments"},
    }

    with comment_col1:
        top_pos_df = filtered_df[filtered_df["pred_sentiment"] == "positive"].sort_values(by='RoBERTa_score', ascending=False)
        top_pos = top_pos_df["review"].head(5)
        style = sentiment_styles["positive"]
        render_comments(style["label"], top_pos, style["bg"], style["text"])

    with comment_col2:
        top_neg_df = filtered_df[filtered_df["pred_sentiment"] == "negative"].sort_values(by='RoBERTa_score', ascending=False)
        top_neg = top_neg_df["review"].head(5)
        style = sentiment_styles["negative"]
        render_comments(style["label"], top_neg, style["bg"], style["text"])



# ============================== EMPTY STATE ==============================
if filtered_df.empty:
    st.warning("No reviews match the selected filters. Try adjusting them.")
