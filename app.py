import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Workforce Readiness AI",
    page_icon="🚀",
    layout="wide"
)

# Load CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

local_css("style.css")

# Define Global Constants (Fixes NameError)
FEATURES = ['Study_Hours', 'Quiz_Scores', 'Attendance', 'Screen_Time', 'Engagement_Score']
RISK_LABELS = {0: "Low (Ready)", 1: "Medium (Monitor)", 2: "High (At-Risk)"}

# 2. Data & Model Loading Logic
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        return pd.read_csv('workforce_data.csv')
    except Exception:
        return None

def load_model():
    try:
        with open('workforce_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

df = load_data()
model = load_model()

# 3. Sidebar Navigation
st.sidebar.title("Workforce Control Tower")
uploaded_file = st.sidebar.file_uploader("Upload New Dataset (CSV)", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)

if df is None or model is None:
    st.error("⚠️ System files missing! Please ensure CSV and PKL files are in the repository.")
    st.stop()

choice = st.sidebar.radio("Navigation", ["Dashboard Overview", "Individual Assessment", "Intervention Strategies"])

# Global Sidebar Filters
st.sidebar.markdown("---")
selected_risks = st.sidebar.multiselect(
    "Filter Dashboard by Risk:", 
    options=list(RISK_LABELS.keys()), 
    default=list(RISK_LABELS.keys()),
    format_func=lambda x: RISK_LABELS.get(x)
)
filtered_df = df[df["Risk_Level"].isin(selected_risks)]

# --- MODULE 1: Dashboard Overview ---
if choice == "Dashboard Overview":
    st.title("🚀 Workforce Performance Analytics")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Professionals", len(filtered_df))
    m2.metric("Avg Technical Score", f"{filtered_df['Quiz_Scores'].mean():.1f}%")
    m3.metric("Avg Engagement", f"{filtered_df['Engagement_Score'].mean():.1f}/10")
    m4.metric("High Risk Count", len(filtered_df[filtered_df["Risk_Level"] == 2]))

    st.markdown("---")
    col_a, col_b = st.columns(2)
    
    with col_a:
        df_pie = filtered_df.copy()
        df_pie["Status"] = df_pie["Risk_Level"].map(RISK_LABELS)
        fig_pie = px.pie(df_pie, names="Status", title="Workforce Health Composition", hole=0.4,
                         color="Status", color_discrete_map={"Low (Ready)": "#00CC96", "Medium (Monitor)": "#FFA15A", "High (At-Risk)": "#EF553B"})
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        fig_hist = px.histogram(filtered_df, x="Quiz_Scores", title="Technical Skill Distribution", color_discrete_sequence=['#3b82f6'])
        st.plotly_chart(fig_hist, use_container_width=True)

    # AI Insights
    st.markdown("---")
    st.subheader("🤖 AI Insights: Risk Drivers")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Metric': FEATURES, 'Importance': importances}).sort_values('Importance')
    fig_imp = px.bar(feat_df, x='Importance', y='Metric', orientation='h', title="Feature Importance Analysis")
    st.plotly_chart(fig_imp, use_container_width=True)

# --- MODULE 2: Individual Assessment ---
elif choice == "Individual Assessment":
    st.subheader("🎯 Predictive Risk Assessment")
    
    with st.form("assessment_form"):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Name", placeholder="John Doe")
            hrs = st.number_input("Monthly Study Hours", 0, 300, 45)
            quiz = st.slider("Technical Quiz Score", 0, 100, 75)
        with c2:
            att = st.slider("Attendance %", 0, 100, 95)
            screen = st.number_input("Daily Screen Time (Hrs)", 0.0, 16.0, 6.5)
            eng = st.slider("Engagement Level (1-10)", 1, 10, 7)
        submit = st.form_submit_button("Run Prediction")

    if submit:
        # Using the Global FEATURES list here (Fixes NameError)
        input_df = pd.DataFrame([[hrs, quiz, att, screen, eng]], columns=FEATURES)
        prediction = model.predict(input_df)[0]
        result_label = RISK_LABELS.get(prediction)

        if prediction == 0: st.success(f"**{name}** is **{result_label}**.")
        elif prediction == 1: st.warning(f"**{name}** is **{result_label}**.")
        else: st.error(f"**{name}** is **{result_label}**.")

# --- MODULE 3: Intervention Strategies ---
elif choice == "Intervention Strategies":
    st.subheader("💡 Prescriptive Intervention Plans")
    target_person = st.selectbox("Select Professional", df["Name"].unique())
    p_data = df[df["Name"] == target_person].iloc[0]

    recs = []
    if p_data['Quiz_Scores'] < 70: recs.append("📚 **Technical Upskilling required.**")
    if p_data['Screen_Time'] > 8: recs.append("🧘 **Fatigue Mitigation: Enforce breaks.**")
    if p_data['Risk_Level'] == 2: recs.append("🚨 **Schedule 1-on-1 review.**")
    
    for r in recs: st.markdown(r)
