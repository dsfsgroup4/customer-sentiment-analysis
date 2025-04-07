# Pour faire tourner le fichier app.py, il faut utiliser le resultats du modèle
# RoBERTa qui se trouve dans le dossier data

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path

# ============================== CONFIG ==============================
DATA_PATH = Path(r"data/model_roberta_results.csv")

# ============================== PAGE SETUP ==============================
st.set_page_config(layout="wide", page_title="Restaurant Review Dashboard", page_icon="📊")

st.markdown("""
    <div style='padding-left: 10px; padding-right: 10px;'>
        <h3>Mcdonald's Dashboard</h3>
    </div>
""", unsafe_allow_html=True)

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
    """Apply sidebar filters and return the filtered DataFrame."""
    st.sidebar.header("Filters")

    start_of_year = pd.Timestamp.now().replace(month=1, day=1)
    max_date = df["review_date"].max()
    min_date = start_of_year
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
    st.session_state["sidebar_date_range"] = date_range

    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df[(df["review_date"] >= start_date) & (df["review_date"] <= end_date)]

    city_options = ["All"] + sorted(df["City"].dropna().unique().tolist())
    location_filter = st.sidebar.selectbox("City", city_options)
    if location_filter != "All":
        df = df[df["City"] == location_filter]

    return df

# ============================== UI HELPERS ==============================
def render_metric(label, value, bg_color, text_color):
    """Render a custom metric box."""
    st.markdown(f"""
        #### {label}
        <div style='background-color:{bg_color}; padding: 0.7rem; border-radius: 6px; text-align:center; font-size:30px; color:{text_color};'>
            {value:,}
        </div>
    """, unsafe_allow_html=True)

def render_comments(title, comments, bg_color, text_color):
    """Render a list of comments in stylized boxes."""
    st.markdown(f"#### {title}")
    for comment in comments:
        st.markdown(f"<div style='background-color:{bg_color}; color:{text_color}; padding:6px; margin-bottom:6px; border-radius:6px; font-size:12px;'>{comment}</div>", unsafe_allow_html=True)

# ============================== MAIN APP ==============================
with st.spinner("Loading data..."):
    df = load_data(DATA_PATH)

if df.empty:
    st.warning("No data available. Please check the source file.")
    st.stop()

filtered_df = apply_filters(df)

dashboard_tab, reviews_tab = st.tabs(["📊 Overview", "📈 Review Trends"])

# ============================== METRICS ==============================
with dashboard_tab:
    total_reviews = len(filtered_df)
    pos_reviews = len(filtered_df[filtered_df["pred_sentiment"] == "positive"])
    neg_reviews = len(filtered_df[filtered_df["pred_sentiment"] == "negative"])
    neu_reviews = len(filtered_df[filtered_df["pred_sentiment"] == "neutral"])

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        render_metric("📊 Total Reviews", total_reviews, "#2E3B4E", "white")
    with metric_col2:
        render_metric("🙂 Positive", pos_reviews, "#1e3d2f", "#b7f7d0")
    with metric_col3:
        render_metric("😠 Negative", neg_reviews, "#3d1e1e", "#ffb6b6")
    with metric_col4:
        render_metric("😐 Neutral", neu_reviews, "#444444", "#eeeeee")

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    st.divider()
# ============================== LOCATION MAP AND WEEKLY TRENDS ==============================
    st.subheader("🗺️ Mcdonald's US Map & Sentiment Trends")
    map_col, trends_col = st.columns([8, 6], gap="small")

    with map_col:
        if "latitude" in filtered_df.columns and "longitude" in filtered_df.columns and "City" in filtered_df.columns:
            location_df = filtered_df.dropna(subset=["latitude", "longitude", "City"])
            location_df = location_df.drop_duplicates(subset=["latitude", "longitude", "City"])
            location_df = filtered_df.dropna(subset=["latitude", "longitude", "City"])
            location_df = location_df.groupby(["latitude", "longitude", "City"]).agg(
                review_count=("clean_reviews", "count")).reset_index()

            city_counts = location_df


            if not city_counts.empty:
                fig_map = px.scatter_geo(
                    city_counts,
                    lat="latitude",
                    lon="longitude",
                    size="review_count",
                    hover_name="City",
                    color="review_count",
                    color_continuous_scale="Reds",
                    scope="usa",
                    template="plotly_dark",
                    title="Restaurants locations",
                    height=400
                )
                fig_map.update_layout(margin=dict(l=0, r=0, t=30, b=10))
                st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("No valid location data available after filtering.")
        else:
            st.info("Location data is unavailable.")


    with trends_col:
        df_trends = filtered_df[filtered_df["review_date"].notna()].copy()

        # === Override trend range to show from beginning of year ===
        start_of_year = pd.Timestamp.now().replace(month=1, day=1)
        max_date = df_trends["review_date"].max().date()
        trend_range = [start_of_year.date(), max_date]

        if len(trend_range) == 2:
            start, end = pd.to_datetime(trend_range[0]), pd.to_datetime(trend_range[1])
            df_trends = df_trends[(df_trends["review_date"] >= start) & (df_trends["review_date"] <= end)]

        df_trends["day"] = df_trends["review_date"].dt.date
        sentiment_counts = df_trends.groupby(["day", "pred_sentiment"]).size().reset_index(name="count")
                
        fig_sentiment = px.line(
            sentiment_counts,
            x="day",
            y="count",
            color="pred_sentiment",
            markers=True,
title=f"Sentiment Trend ({start.strftime('%b %d')} – {end.strftime('%b %d')})",
            
            template="plotly_dark",
            color_discrete_map={"positive": "green", "neutral": "orange", "negative": "red"}
        ,
            hover_data={"count": True})
        st.plotly_chart(fig_sentiment, use_container_width=True, config={'displayModeBar': False})


    st.divider() # line that separates rows

# ============================== TOP COMMENTS ==============================
    st.markdown("### 🗨️ Top Comments")
    comment_col1, comment_col2 = st.columns(2)

    with comment_col1:
        top_pos = filtered_df[filtered_df["pred_sentiment"] == "positive"]["clean_reviews"].head(5)
        render_comments("👍 Top Positive Comments", top_pos, "#1e3d2f", "#b7f7d0")

    with comment_col2:
        top_neg = filtered_df[filtered_df["pred_sentiment"] == "negative"]["clean_reviews"].head(5)
        render_comments("👎 Top Negative Comments", top_neg, "#3d1e1e", "#ffb6b6")

# ============================== EMPTY STATE ==============================
if filtered_df.empty:
    st.warning("No reviews match the selected filters. Try adjusting them.")
