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

#css file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Call the function
local_css("style.css")

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


# 3. Sidebar Navigation & Global State
st.sidebar.title("Workforce Control Tower")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload New Dataset (CSV)", type=["csv"])
df = load_data(uploaded_file)
model = load_model()

# Critical check for required files
if df is None or model is None:
    st.error(
        "⚠️ **System files missing!** Please ensure 'workforce_data.csv' and 'workforce_model.pkl' exist in your directory.")
    st.stop()

choice = st.sidebar.radio(
    "Navigation",
    ["Dashboard Overview", "Individual Assessment", "Intervention Strategies"]
)

# Global Sidebar Filters
st.sidebar.markdown("---")
st.sidebar.subheader("Live Filters")
risk_options = sorted(df["Risk_Level"].unique())
risk_map_labels = {0: "Low (Ready)", 1: "Medium (Monitor)", 2: "High (At-Risk)"}
selected_risks = st.sidebar.multiselect(
    "Filter Dashboard by Risk:",
    risk_options,
    default=risk_options,
    format_func=lambda x: risk_map_labels.get(x)
)

filtered_df = df[df["Risk_Level"].isin(selected_risks)]

# 4. Main Interface
st.title("🚀 Intelligent System for Workforce Readiness")
st.markdown("Automated analytics for performance stability and fatigue management.")

# --- MODULE 1: Dashboard Overview ---
if choice == "Dashboard Overview":
    # Row 0: KPI Metrics
    m1, m2, m3, m4 = st.columns(4)
    avg_quiz = filtered_df["Quiz_Scores"].mean()
    avg_eng = filtered_df["Engagement_Score"].mean()
    high_risk_count = len(filtered_df[filtered_df["Risk_Level"] == 2])

    m1.metric("Total Professionals", len(filtered_df))
    m2.metric("Avg Technical Score", f"{avg_quiz:.1f}%")
    m3.metric("Avg Engagement", f"{avg_eng:.1f}/10")
    m4.metric("High Risk Count", high_risk_count, delta_color="inverse")

    st.markdown("---")

    # Row 1: Charts
    col_a, col_b = st.columns(2)

    with col_a:
        # Donut Chart for Risk Distribution
        df_pie = filtered_df.copy()
        df_pie["Status"] = df_pie["Risk_Level"].map(risk_map_labels)
        fig_pie = px.pie(
            df_pie, names="Status",
            title="Workforce Health Composition",
            hole=0.4,
            color="Status",
            color_discrete_map={
                "Low (Ready)": "#00CC96",
                "Medium (Monitor)": "#FFA15A",
                "High (At-Risk)": "#EF553B"
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        # Technical Proficiency Histogram
        fig_hist = px.histogram(
            filtered_df, x="Quiz_Scores", nbins=15,
            title="Distribution of Technical Skills",
            color_discrete_sequence=['#636EFA']
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Row 2: AI Insights (Feature Importance)
    st.markdown("---")
    st.subheader("🤖 AI Insights: Risk Drivers")

    features = ['Study_Hours', 'Quiz_Scores', 'Attendance', 'Screen_Time', 'Engagement_Score']
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Metric': features, 'Importance': importances}).sort_values('Importance')

    col_feat1, col_feat2 = st.columns([2, 1])
    with col_feat1:
        fig_imp = px.bar(feat_df, x='Importance', y='Metric', orientation='h',
                         title="Impact of Metrics on Prediction", color='Importance')
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_feat2:
        top_driver = feat_df.iloc[-1]['Metric']
        st.info(
            f"**AI Findings:** The model indicates that **{top_driver}** is the strongest predictor of workforce risk. Management should focus on optimizing this area.")

# --- MODULE 2: Individual Assessment ---
elif choice == "Individual Assessment":
    st.subheader("🎯 Real-Time Predictive Assessment")
    st.write("Input professional data to calculate readiness and fatigue risk.")

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
        # Create input for model
        input_df = pd.DataFrame([[hrs, quiz, att, screen, eng]], columns=features)
        prediction = model.predict(input_df)[0]
        result_label = risk_map_labels.get(prediction)

        st.markdown("### Assessment Result")
        if prediction == 0:
            st.success(f"**{name}** is **{result_label}**.")
        elif prediction == 1:
            st.warning(f"**{name}** is **{result_label}**.")
        else:
            st.error(f"**{name}** is **{result_label}**.")

        # Export Feature
        report = f"Assessment Report\nName: {name}\nStatus: {result_label}\nScores: Q-{quiz}% | E-{eng}/10"
        st.download_button("📥 Download Report", report, f"Report_{name}.txt")

# --- MODULE 3: Intervention Strategies ---
elif choice == "Intervention Strategies":
    st.subheader("💡 Prescriptive Intervention Plans")

    target_person = st.selectbox("Select Professional to Review", df["Name"].unique())
    p_data = df[df["Name"] == target_person].iloc[0]

    st.markdown(f"### Strategy for {target_person}")
    recs = []

    if p_data['Quiz_Scores'] < 70:
        recs.append("📚 **Technical Upskilling:** Enroll in peer-mentoring sessions.")
    if p_data['Screen_Time'] > 8:
        recs.append("🧘 **Fatigue Mitigation:** Enforce 15-minute screen breaks every 2 hours.")
    if p_data['Risk_Level'] == 2:
        recs.append("🚨 **High Priority:** Schedule a direct 1-on-1 performance review.")

    if not recs:
        st.success("Candidate is performing optimally. Maintain standard monitoring.")
        recs.append("Maintain current path.")
    else:
        for r in recs: st.markdown(r)

    st.download_button("📥 Export Plan", "\n".join(recs), f"Plan_{target_person}.txt")