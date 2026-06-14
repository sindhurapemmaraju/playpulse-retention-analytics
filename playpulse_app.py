import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="PlayPulse", page_icon="🎮", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("clean_apps.csv")
    insights = pd.read_csv("cluster_insights.csv")
    return df, insights

df, insights_df = load_data()

st.markdown("<h1 style='color:#1A3A5C; text-align:center;'>PlayPulse</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='color:#2E75B6; text-align:center;'>Google Play Store Retention Intelligence Pipeline</h4>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>9,658 apps — EDA · Hypothesis Testing · ML (ROC-AUC 0.93) · Clustering · Gemini AI</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.markdown("<h3 style='text-align:center'>9,658</h3><p style='text-align:center'>Apps Analyzed</p>", unsafe_allow_html=True)
col2.markdown("<h3 style='text-align:center'>4.19</h3><p style='text-align:center'>Avg Rating</p>", unsafe_allow_html=True)
col3.markdown("<h3 style='text-align:center'>0.93</h3><p style='text-align:center'>ROC-AUC Score</p>", unsafe_allow_html=True)
col4.markdown("<h3 style='text-align:center'>6</h3><p style='text-align:center'>App Segments</p>", unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["EDA", "Hypothesis Tests", "ML Results", "Clusters and AI Insights"])

with tab1:
    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        top_cats = df.groupby("Category").agg(App_Count=("App","count"), Avg_Rating=("Rating","mean")).sort_values("App_Count", ascending=False).head(15).reset_index()
        fig1 = px.bar(top_cats, x="Category", y="App_Count", color="Avg_Rating", color_continuous_scale="Blues", title="Top 15 Categories by App Count")
        fig1.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(df, x="Rating", nbins=30, color_discrete_sequence=["#2E75B6"], title="Rating Distribution")
        fig2.add_vline(x=df["Rating"].mean(), line_dash="dash", line_color="red", annotation_text=f"Mean: {df['Rating'].mean():.2f}")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        type_counts = df["Type"].value_counts().reset_index()
        fig3 = px.pie(type_counts, names="Type", values="count", color_discrete_map={"Free":"#2E75B6","Paid":"#1A3A5C"}, title="Free vs Paid Apps")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        df["Category_Type"] = df["Is_Indian_Category"].map({1:"Indian-Relevant", 0:"Global"})
        fig4 = px.box(df, x="Category_Type", y="Rating", color="Category_Type", color_discrete_map={"Indian-Relevant":"#FF6B35","Global":"#2E75B6"}, title="Indian-Relevant vs Global Ratings")
        st.plotly_chart(fig4, use_container_width=True)

with tab2:
    st.subheader("Statistical Hypothesis Testing")
    st.markdown("""
| Test | Finding | p-value | Practical Significance |
|------|---------|---------|----------------------|
| Free vs Paid | Paid apps rated slightly higher (4.27 vs 4.18) | p < 0.0001 | Negligible — 0.09 diff, identical medians |
| Indian vs Global | No meaningful difference (4.18 vs 4.19) | p = 0.30 | Negligible — Cohen d = -0.016 |
| Update recency | Recently updated apps rated higher | p < 0.0001 | Weak but consistent (r = -0.116) |
    """)
    st.info("Key Insight: Statistical significance is not practical significance. Ratings are driven by update recency, not price or cultural category.")
    df["Update_Recency"] = pd.cut(df["Days_Since_Update"], bins=[0,30,90,180,365,9999], labels=["<1 Month","1-3 Months","3-6 Months","6-12 Months","1+ Year"])
    recency = df.groupby("Update_Recency", observed=True)["Rating"].mean().reset_index()
    fig5 = px.bar(recency, x="Update_Recency", y="Rating", color="Rating", color_continuous_scale="Blues", title="Rating by Update Recency", text="Rating")
    fig5.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig5.update_layout(yaxis=dict(range=[3.8, 4.4]))
    st.plotly_chart(fig5, use_container_width=True)

with tab3:
    st.subheader("Machine Learning — Retention Classification")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model Comparison")
        results = pd.DataFrame({
            "Metric": ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"],
            "Logistic Regression": [0.8385, 0.8456, 0.8506, 0.8481, 0.9172],
            "Random Forest": [0.8618, 0.8199, 0.9473, 0.8790, 0.9308]
        })
        st.dataframe(results, use_container_width=True)
        st.success("Random Forest selected — ROC-AUC 0.93, Recall 0.947")

    with col2:
        st.markdown("### Feature Importance")
        importance = pd.DataFrame({
            "Feature": ["Log_Installs","Log_Reviews","Days_Since_Update","Size_MB","Category_Encoded","Price","Content_Rating_Encoded","Is_Indian_Category","Type_Encoded"],
            "Importance": [0.356, 0.334, 0.108, 0.089, 0.067, 0.022, 0.014, 0.006, 0.004]
        })
        fig6 = px.bar(importance, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Blues", title="What Drives Retention?")
        fig6.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig6, use_container_width=True)

    st.info("Regression R2 = 0.12 but Classification ROC-AUC = 0.93. Install volume and review count are the dominant retention signals.")

with tab4:
    st.subheader("K-Means Clustering — 6 App Segments")
    cluster_summary = pd.DataFrame({
        "Segment": ["Mainstream Champions","Hidden Gems","Viral Giants","Struggling Survivors","Abandoned but Alive","Premium Outliers"],
        "App Count": [3606, 2998, 1114, 910, 1012, 18],
        "Avg Rating": [4.30, 4.38, 4.27, 3.08, 4.13, 3.93]
    })
    fig7 = px.bar(cluster_summary, x="Segment", y="App Count", color="Avg Rating", color_continuous_scale="Blues", title="App Segments — Size and Quality", text="App Count")
    fig7.update_traces(textposition="outside")
    fig7.update_layout(xaxis_tickangle=-20)
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("### Gemini AI — Cluster Insights")
    for _, row in insights_df.iterrows():
        with st.expander(row["Cluster_Name"]):
            st.write(row["AI_Insight"])

    st.caption("Insights generated using Gemini 2.5 Flash API via prompt engineering on cluster profile data.")
